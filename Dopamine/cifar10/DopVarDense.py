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

class DopVarDense(Layer):

	def __init__(self, activation, units=32, threshold=0, learning_rate=None):
		'''Initializes the instance attributes'''
		super(DopVarDense, self).__init__()
		self.activation = tf.keras.activations.get(activation)
		self.units = units
		self.threshold = threshold
		self.lr = learning_rate


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
		self.hebb = tf.Variable(tf.zeros([input_shape[-1], self.units], tf.float32), trainable=False)

		
		
	def call(self, inputs):
	
		'''Defines the computation from inputs to outputs'''
		
		outputs = tf.matmul(inputs, self.w)
		
		tf.print(inputs.shape)
		tf.print(outputs.shape, "\n\n")
		
		
		
		
		#if(tf.keras.backend.in_train_phase(1, 0)):
		
			#tf.print("\n\n\nepoch: ", epoch_variable, "batch: ", self.batch_cntr)
		
		#	if(self.batch_cntr > 1):
			
				
				
				


	
		
		tf.keras.backend.update_add(self.batch_cntr, 1)
		tf.keras.backend.update(self.hebb, self.w)
		

		return self.activation(tf.matmul(inputs, self.w) + self.b)







