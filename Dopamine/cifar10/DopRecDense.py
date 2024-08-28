import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Input, Layer, Dense
from matplotlib import pyplot


epoch_variable = np.array(0, dtype=np.int32)
cnt = np.zeros((1500))

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

class DopRecDense(Layer):

	def __init__(self, activation, units=32, threshold=0, learning_rate=None):
		'''Initializes the instance attributes'''
		super(DopRecDense, self).__init__()
		self.units = units
		self.threshold = threshold
		self.activation = tf.keras.activations.get(activation)
		self.lr = learning_rate

		
		## Initialize refraction period indicator for each dopaminergic neuron
		#self.indicator = -self.ref_period*np.ones((self.units))

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
		self.weights_old = tf.Variable(tf.zeros([input_shape[-1], self.units], tf.float32), trainable=False)
		self.prev_weights_old = tf.Variable(tf.zeros([2304, 512], tf.float32), trainable=False)
		
		## Dopaminergic neuron filter
		f = np.zeros((self.units, input_shape[-1], self.units), dtype=np.float32)
		for i in range(self.units):
			f[i,:,i] = 1
		self.filter = tf.Variable(f, dtype=tf.float32, trainable=False)
		
	def d2_sig(self,a,b,x):
		
		return 1/(1+tf.math.exp(-a*(x-b)))
		
	def d1_sig(self,a,b,x):
		
		return -1/(1+tf.math.exp(-a*(x-b))) + 1
		
		
	def call(self, inputs):
	
		'''Defines the computation from inputs to outputs'''
		
		n_inputs = np.shape(inputs)[-1]
		
		if(tf.keras.backend.in_train_phase(1, 0)):
		
			#tf.print("\n\n\nepoch: ", epoch_variable, "batch: ", self.batch_cntr)
		
			if(self.batch_cntr > 1):
			
				pos_mean_difference = tf.math.reduce_mean( tf.keras.activations.relu( tf.math.subtract(self.w, self.weights_old) ), axis=0)
				neg_mean_difference = tf.math.reduce_mean( -tf.keras.activations.relu( -tf.math.subtract(self.w, self.weights_old) ), axis=0)
				
				d1_factor = self.d1_sig(100, 0.02, pos_mean_difference)
				d2_factor = self.d2_sig(100, 0.02, pos_mean_difference)

				#tf.print("\nmean_differences\n", pos_mean_difference, summarize=-1)
				#print(" D1: ", d1_factor, "\n\n")
				#print(" D2: ", d2_factor, "\n\n")

				#tf.print("\nweights:\n", self.w)

				for i in range(self.units):
				
					k = i+1
					if(k == self.units):
						k = 0
				
					#if(self.batch_cntr-self.indicator[i] > self.ref_period):
					
					left = d1_factor[i]*pos_mean_difference[i-1] + d2_factor[i]*neg_mean_difference[i-1]
					right = d1_factor[i]*pos_mean_difference[k] + d2_factor[i]*neg_mean_difference[k]
					
					aux_factor = 0.1*( left *self.filter[i] + right *self.filter[i])
					
					#print("\n\n", i, " Left: ", left)
					#print("\n", i, " Right: ", right, "\n\n")
					
					(self.w).assign(self.w + aux_factor)
	
	
		
			tf.keras.backend.update_add(self.batch_cntr, 1)
			
			tf.keras.backend.update(self.weights_old, self.w)
		

		return self.activation(tf.matmul(inputs, self.w) + self.b)




















