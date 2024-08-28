import tensorflow as tf
import numpy as np
import random

tf.config.run_functions_eagerly(True)

from tensorflow.keras.callbacks import Callback
from itertools import combinations_with_replacement
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, Dense
from matplotlib import pyplot

from sklearn import datasets
from sklearn.model_selection import train_test_split


iris = datasets.load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

print("\nDataset size: ", np.shape(x), "\n")




class MyCallback(Callback):
	def __init__(self, epoch_variable):
		self.epoch_variable = epoch_variable
		
	def on_epoch_begin(self, epoch, logs=None):
		self.epoch_variable += 1
	'''	
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
	'''
class DopVarDense(Layer):

	def __init__(self, activation, units=32, n_dop=None, threshold=0, refractory_period=0):
		'''Initializes the instance attributes'''
		super(DopVarDense, self).__init__()
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

		## Initialize refraction period indicator for each dopaminergic neuron
		self.indicator = -self.ref_period*np.ones((self.units))

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
		self.dop_weights_old = tf.Variable(tf.zeros([input_shape[-1], self.units], tf.float32), trainable=False)

		
		
	def call(self, inputs):
		'''Defines the computation from inputs to outputs'''
		
		n_inputs = np.shape(inputs)[-1]
		
		## Dopaminergic neuron filter
		#f = np.zeros((self.n_dop, n_inputs, self.units), dtype=np.float32)
		#for i in range(self.n_dop):
		#	f[i,:,self.dop_indices[i]] = 1
		#self.filter = tf.Variable(f, dtype=tf.float32, trainable=False)
		
		
		
		
		if(tf.keras.backend.in_train_phase(1, 0)):
		
			#Each epoch has a number of batches depending on batch size and the size of the dataset to go through (batch iterations = dataset size / batch size)
			tf.print("\n\n\nepoch: ", epoch_variable, "batch: ", self.batch_cntr)
		
			
			weight_multiplier = np.zeros((n_inputs, self.units), dtype=np.float32)
			
			
			if(self.batch_cntr > 0):
			
				tf.print("\nold:\n", self.dop_weights_old, summarize=-1)
				tf.print("\nnew:\n", self.w, summarize=-1)
				
				#Calculate the average difference in connection weights from t-1 to t at each dopaminergic neuron - How much the connection changed
				dop_neuron_mean_difference = tf.math.reduce_sum( tf.math.abs( tf.math.subtract(self.w, self.dop_weights_old) ), axis=0) / n_inputs

				#Go through all neuron cell
				for i in range(self.units):
				
					#Check if the average difference in this specific cell is over a threshold and if the cell is not under refractory period
					if( dop_neuron_mean_difference[i] > self.threshold  and self.batch_cntr-self.indicator[i] > self.ref_period):
						
						aux_factor = np.zeros((n_inputs, self.units), dtype=np.float32)
						
						
						if(not all(weight_multiplier[:,i-1])):
							aux_factor[:,i-1] = 1
						
						k = i+1
						if(k==self.units):
							k = -1	
						if(not all(weight_multiplier[:,k+1])):
							aux_factor[:,i+1] = 1
						
						
						
						weight_multiplier = weight_multiplier + aux_factor
						
						
						tf.print("\nDopNeuron ", i, " is activating with delta ", dop_neuron_mean_difference[i], ".\n")
						
						
						## Register that this dopaminergic neuron just activated on this batch
						self.indicator[i] = self.batch_cntr
		
				
				
				tf.print("\nwm:\n", weight_multiplier, summarize=-1)
				tf.print("\n1-wm:\n", 1-weight_multiplier, summarize=-1)
				
				
				
				
				(self.w).assign( tf.math.multiply(self.w,weight_multiplier) + tf.math.multiply(self.dop_weights_old,(1-weight_multiplier)) )
			tf.print("\nweights_old\n", self.dop_weights_old, summarize=-1)
			tf.print("\nweights:\n", self.w, summarize=-1)
				
			tf.keras.backend.update(self.dop_weights_old, self.w)	
			tf.keras.backend.update_add(self.batch_cntr, 1)
			
			
			
		
		return self.activation(tf.matmul(inputs, self.w) + self.b)

			
		
		

opt = tf.keras.optimizers.Adam(lr=0.001)

i = Input((4,))
h = Dense(5, activation='relu')(i)
o = Dense(3, activation='softmax')(h)
model = Model(inputs=i, outputs=o)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

epoch_variable = np.array(0, dtype=np.int32)

inp = Input((4,))
h1 = DopVarDense('relu', 5, 2, 0, 0)(inp)
out = Dense(3, activation='softmax')(h1)
dop_model = Model(inputs=inp, outputs=out)
dop_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


dop_history = dop_model.fit(x_train, y_train, epochs=50, verbose=1, batch_size=4, callbacks=[MyCallback(epoch_variable)], shuffle=True, validation_split=0.2)  
history = model.fit(x_train, y_train, epochs=50, verbose=1, batch_size=4, shuffle=True)

tf.keras.backend.set_learning_phase(0)
dop_results = dop_model.evaluate(x_test, y_test, batch_size=32)
print("Dopaminergic test loss: ", dop_results[0])
print("Dopaminergic test acc: ", dop_results[1])

results = model.evaluate(x_test, y_test, batch_size=32)
print("Conventional test loss: ", results[0])
print("Conventional test acc: ", results[1])


#fig, ((ax1, ax2), (ax3, ax4)) = pyplot.subplots(2, 2)
fig, ((ax1, ax3)) = pyplot.subplots(2, 1)

#ax1.plot(dop_history.history[activations])


ax1.plot(history.history['accuracy'])
ax1.plot(dop_history.history['accuracy'])
ax1.set_title('model accuracy')
ax1.set_ylabel('accuracy')
ax1.set_xlabel('epoch')
ax1.legend(['Conventional', 'Dopaminergic'], loc='lower right')

'''
ax2.plot(history.history['val_accuracy'])
ax2.plot(dop_history.history['val_accuracy'])
ax2.set_title('validation accuracy')
ax2.set_ylabel('accuracy')
ax2.set_xlabel('epoch')
ax2.legend(['Conventional', 'Dopaminergic'], loc='lower right')
'''

ax3.plot(history.history['loss'])
ax3.plot(dop_history.history['loss'])
ax3.set_title('model loss')
ax3.set_ylabel('loss')
ax3.set_xlabel('epoch')
ax3.legend(['Conventional', 'Dopaminergic'], loc='upper right')

'''
ax4.plot(history.history['val_loss'])
ax4.plot(dop_history.history['val_loss'])
ax4.set_title('validation loss')
ax4.set_ylabel('loss')
ax4.set_xlabel('epoch')
ax4.legend(['Convetional', 'Dopaminergic'], loc='upper right')
'''

pyplot.show()

















