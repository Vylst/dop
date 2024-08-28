import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Input, Layer, Dense
from matplotlib import pyplot


epoch_variable = np.array(0, dtype=np.int32)
cnt = np.zeros((300))

class MyCallback(Callback):
	def __init__(self):
		self.epoch_variable = epoch_variable
		
	def on_epoch_begin(self, epoch, logs=None):
		self.epoch_variable += 1
		
	def on_training_end(self, logs=None):
		print("Dopaminergic activations: \n")
		print('[', end='')
		for i in range(len(cnt)):
			print(cnt[i], ',', end='')
		print(']')
		
		ax = pyplot.subplot(1,1,1)
		ax.plot(cnt[0:150], color='green')
		ax.set_title('Number of Dopaminergic Activations')
		ax.set_ylabel('Activations')
		ax.set_xlabel('epoch')
		ax.legend(['Activations'], loc='upper right')

class DopDense(Layer):

	def __init__(self, activation, units=32, n_dop=None, threshold=0, refractory_period=0):
		'''Initializes the instance attributes'''
		super(DopDense, self).__init__()
		self.units = units
		self.threshold = threshold
		self.ref_period = refractory_period
		self.activation = tf.keras.activations.get(activation)
		
		if(n_dop == None):
			self.n_dop = np.random.randint(1, units/2)
		
		else:
			if(n_dop <= units/2):			
				self.n_dop = n_dop	
			else:
				raise ValueError("Dopaminergic neurons must not exceed half of layer size.")

		## Generate indices for dopaminergic neurons
		self.dop_indices = np.linspace(1, self.units-1, self.n_dop, dtype=np.int32)

		## Initialize refraction period indicator for each dopaminergic neuron
		self.indicator = -self.ref_period*np.ones((self.n_dop))

		## Batch counter
		self.batch_cntr = tf.Variable(0, dtype='int64', name='counter', trainable=False)

	def build(self, input_shape):
		'''Create the state of the layer (weights)'''

		## initialize the weights
		w_init = tf.random_normal_initializer()
		self.w = tf.Variable(name="kernel", initial_value=w_init(shape=(input_shape[-1], self.units), dtype='float32'), trainable=True)

		## initialize the biases
		b_init = tf.zeros_initializer()
		self.b = tf.Variable(name="bias", initial_value=b_init(shape=(self.units,), dtype='float32'), trainable=True)
		
		## Placeholders for dopaminergic neuron weights
		self.dop_weights_new = tf.Variable(tf.zeros([input_shape[-1], self.units], tf.float32), trainable=False)
		self.dop_weights_old = tf.Variable(tf.zeros([input_shape[-1], self.units], tf.float32), trainable=False)
		
		## Dopaminergic neuron filter
		f = np.zeros((self.n_dop, input_shape[-1], self.units), dtype=np.float32)
		for i in range(self.n_dop):
			f[i,:,self.dop_indices[i]] = 1
		self.filter = tf.Variable(f, dtype=tf.float32, trainable=False)
		
	def call(self, inputs):
		'''Defines the computation from inputs to outputs'''
		
		n_inputs = np.shape(inputs)[-1]
		
		if(tf.keras.backend.in_train_phase(1, 0)):
		
			tf.print("Dopaminergic neurons: ", self.dop_indices, "\n")
		
			## Gather weights from dopaminergic neurons
			(self.dop_weights_new).assign(tf.math.multiply(self.w, tf.reduce_sum(self.filter,axis=0)))
		
			weight_multiplier = tf.Variable(np.ones((n_inputs, self.units), dtype=np.float32), trainable=False)
			if(self.batch_cntr > 1):
			
				dop_neuron_mean_difference = tf.math.reduce_sum( tf.math.abs( tf.math.subtract(self.dop_weights_new, self.dop_weights_old) ), axis=0) / n_inputs
		
				c = 0
				for i in range(self.n_dop):
				
					## For each dopaminergic neuron, confirm if its temporal delta is greater than threshold and also if the neuron is within a refractory period 
					if( dop_neuron_mean_difference[self.dop_indices[i]] > self.threshold  and self.batch_cntr-self.indicator[i] > self.ref_period):
						
						## Make sure the rolled filter does not encompass neighboring dopaminergic neurons (e.g. dop neuron 10 can't impact dop neuron 11 and vice-versa)
						aux_factor = 0;
						if(self.dop_indices[i]-1 not in self.dop_indices):
							aux_factor += tf.roll(self.filter[i],-1,1)
						if(self.dop_indices[i]+1 not in self.dop_indices):
							aux_factor += tf.roll(self.filter[i],1,1)
						
						weight_multiplier = weight_multiplier * ( 1 + (10*dop_neuron_mean_difference[self.dop_indices[i]]) * aux_factor )
						
						tf.print("\naux:\n", aux_factor, summarize=-1)
						
						## Register that this dopaminergic neuron just activated on this batch
						self.indicator[i] = self.batch_cntr
						c = c + 1
		
				cnt[epoch_variable-1] = cnt[epoch_variable-1] + c
		
			#tf.print("\n\n\nepoch: ", epoch_variable, " batch: ", self.batch_cntr)
			#tf.print("\nweights:\n", self.w, summarize=-1)
			#tf.print("\nwm:\n", weight_multiplier, summarize=-1)
		
		
			tf.keras.backend.update_add(self.batch_cntr, 1)
			tf.keras.backend.update(self.dop_weights_old, self.dop_weights_new)
			(self.w).assign(tf.math.multiply(self.w,weight_multiplier))
		
			#tf.print("\nweights:\n", self.w, summarize=-1)
		
		return self.activation(tf.matmul(inputs, self.w) + self.b)




















