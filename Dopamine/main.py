import tensorflow as tf
import numpy as np
import random

tf.config.run_functions_eagerly(True)

from tensorflow.keras.callbacks import Callback
from itertools import combinations_with_replacement
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from matplotlib import pyplot

from sklearn import datasets
from sklearn.model_selection import train_test_split


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


class MyCallback(Callback):
	def __init__(self, epoch_variable):
		self.epoch_variable = epoch_variable

	def on_epoch_begin(self, epoch, logs=None):
		#tf.keras.backend.set_value(self.epoch_variable, epoch + 1)
		self.epoch_variable += 1

class DopDense(Layer):

	def __init__(self, activation, units=32, n_dop=None, threshold=0, refractory_period=0):
		'''Initializes the instance attributes'''
		super(DopDense, self).__init__()
		self.units = units
		self.threshold = threshold
		self.ref_period = refractory_period
		self.activation = activation
		
		if(n_dop == None):
			self.n_dop = np.random.randint(1, units/2)
		
		else:
			if(n_dop <= units/2):			
				self.n_dop = n_dop	
			else:
				raise ValueError("Dopaminergic neurons must not exceed half of layer size.")

		## Generate indices for dopaminergic neurons
		#self.dop_indices = random.sample(range(1, self.units-1), self.n_dop)
		#(self.dop_indices).sort()
		self.dop_indices = np.linspace(1, self.units-1, self.n_dop, dtype=np.int32)

		## Placeholders for dopaminergic neuron weights
		self.dop_weights_new = tf.Variable(0., trainable=False)
		self.dop_weights_old = tf.Variable(0., trainable=False)

		## Initialize refraction period indicator for each dopaminergic neuron
		self.indicator = -self.ref_period*np.ones((self.n_dop))


	def build(self, input_shape):
		'''Create the state of the layer (weights)'''        

		self.batch_ctr = 0#np.array(0, dtype=np.int32)
		

		## Number of connections per neuron in this layer, with neurons from the previous layer
		self.n_inputs = input_shape[-1]

		## initialize the weights
		w_init = tf.random_normal_initializer()
		self.w = tf.Variable(name="kernel", initial_value=w_init(shape=(input_shape[-1], self.units), dtype='float32'), trainable=True)

		## initialize the biases
		b_init = tf.zeros_initializer()
		self.b = tf.Variable(name="bias", initial_value=b_init(shape=(self.units,), dtype='float32'), trainable=True)
		
		## Dopaminergic neuron filter
		f = np.zeros((self.n_dop, input_shape[-1], self.units), dtype=np.float32)
		for i in range(self.n_dop):
			f[i,:,self.dop_indices[i]] = 1
		self.filter = tf.Variable(f, dtype=tf.float32, trainable=False)
		
		'''
		tf.print("input shape", input_shape[-1])
		tf.print("Weight shape: ", np.shape(self.w))
		'''

	## Check if training or testing
	def train_test(self):
		return tf.keras.backend.in_train_phase(1, 0)
	

	def call(self, inputs):
		'''Defines the computation from inputs to outputs'''

		print("\nlearning phase: ", self.train_test())


		if(self.train_test()):
			## Gather weights from dopaminergic neurons
			self.dop_weights_new = tf.math.multiply(self.w, tf.reduce_sum(self.filter,axis=0))
			
			
			#tf.print("\n\n\nepoch: ", epoch_variable, "batch: ", self.batch_ctr)
			#tf.print("Dopaminergic neurons: ", self.dop_indices, "\n")
			#tf.print("Number of dop neurons: ", self.n_dop, "\n")
			#tf.print("\nweights:\n", weights, summarize=-1)
			
			weight_multiplier = tf.Variable(np.ones((np.shape(inputs)[-1], self.units), dtype=np.float32), trainable=False)
			if(self.batch_ctr > 1):
				
				## Compute differences in dopaminergic weights between batch t and t-1, and average by number of dopaminergic connections per dopaminergic neuron
				dop_neuron_mean_difference = tf.math.reduce_sum( tf.math.abs( tf.math.subtract(self.dop_weights_new, self.dop_weights_old) ), axis=0) / self.n_inputs
				

				tf.print("\nweights_new\n", self.dop_weights_new, summarize=-1)
				tf.print("\nweights_old\n", self.dop_weights_old, summarize=-1)
				#tf.print("\ndifferences\n", tf.math.subtract(self.dop_weights_new, self.dop_weights_old), summarize=-1)
				tf.print("\nmean_differences\n", dop_neuron_mean_difference, summarize=-1)

				
				for i in range(self.n_dop):
				
					#tf.print(self.batch_ctr-self.indicator[i], " vs ", self.ref_period)
					#tf.print(dop_neuron_mean_difference[self.dop_indices[i]], " vs ", self.threshold)
				
					## For each dopaminergic neuron, confirm if its temporal delta is greater than threshold and also if the neuron is within a refractory period 
					if( dop_neuron_mean_difference[self.dop_indices[i]] > self.threshold  and self.batch_ctr-self.indicator[i] > self.ref_period):
				
						## Make sure the rolled filter does not encompass neighboring dopaminergic neurons (e.g. dop neuron 10 can't impact dop neuron 11 and vice-versa)
						aux_factor = 0;
						if(self.dop_indices[i]-1 not in self.dop_indices):
							aux_factor += tf.roll(self.filter[i],-1,1)
						if(self.dop_indices[i]+1 not in self.dop_indices):
							aux_factor += tf.roll(self.filter[i],1,1)
						
						weight_multiplier = weight_multiplier * ( 1 + (10*dop_neuron_mean_difference[self.dop_indices[i]]) * aux_factor )
						
						tf.print("\nDopNeuron ", self.dop_indices[i], " is activating with delta ", dop_neuron_mean_difference[self.dop_indices[i]], ".\n")
						
						
						#tf.print("\nAux_factor: \n", aux_factor, summarize=-1 )
						#tf.print("\nwm:\n", weight_multiplier, summarize=-1)
						#tf.print("\nindicator: ", self.indicator[i], "  epoch_variable: ", epoch_variable)
						
						
						## Register that this dopaminergic neuron just activated on this batch
						self.indicator[i] = self.batch_ctr
						

			
			## Store current weights as old weights for next batch
			self.dop_weights_old = self.dop_weights_new
			
			## Increment batch counter
			self.batch_ctr = self.batch_ctr + 1
			
			(self.w).assign(self.w*weight_multiplier)
		
		## Usual linear combination of fully connected layers
		if(self.activation == "relu"):
			return tf.keras.activations.relu(tf.matmul(inputs, self.w) + self.b)
		elif(self.activation == "softmax"):
			return tf.keras.activations.softmax(tf.matmul(inputs, self.w) + self.b)
		else:
			return tf.matmul(inputs, self.w) + self.b
	
'''
L = [];	
A = [];
opt = tf.keras.optimizers.Adam(learning_rate=0.001)		
for i in range(150):

	print("\nEvaluating Dop layer with ", i, " dopaminergic neurons.\n")

	tf.keras.backend.clear_session()
	epoch_variable = np.array(0, dtype=np.int32)

	inp = Input(input_shape)
	h1 = Flatten()(inp)
	h2 = DopDense(1024, i, 0, 0, activation='relu')(h1)
	out = DopDense(num_classes, 4, 0, 0, activation='softmax')(h2)
	dop_model = Model(inputs=inp, outputs=out)
	dop_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	
	dop_model.summary()
	
	dop_history = dop_model.fit(x_train, y_train, epochs=15, batch_size=128, verbose=1, callbacks=[MyCallback(epoch_variable)])  

	[loss, accuracy] = dop_model.evaluate(x_test, y_test)
	L = L + [loss]
	A = A + [accuracy]
	
fig, ((ax1, ax2)) = pyplot.subplots(1, 2)

ax1.plot(A)
ax1.set_title('Accuracy')
ax1.set_ylabel('accuracy')
ax1.set_xlabel('Dopaminergic Neurons')

ax2.plot(L)
ax2.set_title('Loss')
ax2.set_ylabel('loss')
ax2.set_xlabel('Dopaminergic Neurons')

fig.savefig("teste.png")

pyplot.show()
'''




opt = tf.keras.optimizers.Adam(lr=0.0001) #0.001 is default for adam
			
i = Input(input_shape)
h_1 = Conv2D(32, kernel_size=(3, 3), activation="relu")(i)
h_2 = MaxPooling2D(pool_size=(2, 2))(h_1)
h_3 = Conv2D(64, kernel_size=(3, 3), activation="relu")(h_2)
h_4 = MaxPooling2D(pool_size=(2, 2))(h_3)
h_5 = Flatten()(h_4)
h_6 = Dropout(0.5)(h_5)
o = Dense(num_classes, activation="softmax")(h_6)
model = Model(inputs=i, outputs=o)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

epoch_variable = np.array(0, dtype=np.int32)
inp = Input(input_shape)
h1 = Conv2D(32, kernel_size=(3, 3), activation="relu")(inp)
h2 = MaxPooling2D(pool_size=(2, 2))(h1)
h3 = Conv2D(64, kernel_size=(3, 3), activation="relu")(h2)
h4 = MaxPooling2D(pool_size=(2, 2))(h3)
h5 = Flatten()(h4)
h6 = Dropout(0.5)(h5)
out = DopDense("softmax", num_classes, 2, 0.0001, 0)(h6)
dop_model = Model(inputs=inp, outputs=out)
dop_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

dop_model.summary()

dop_history = dop_model.fit(x_train, y_train, epochs=5, batch_size=128, verbose=1, callbacks=[MyCallback(epoch_variable)])
history = model.fit(x_train, y_train, epochs=5, batch_size=128, verbose=1)



results = model.evaluate(x_test, y_test, batch_size=128)
print("Conventional test loss: ", results[0])
print("Conventional test acc: ", results[1])

dop_results = dop_model.evaluate(x_test, y_test, batch_size=128)
print("Dopaminergic test loss: ", dop_results[0])
print("Dopaminergic test acc: ", dop_results[1])


fig, ((ax1, ax3)) = pyplot.subplots(2, 1)

ax1.plot(history.history['accuracy'])
ax1.plot(dop_history.history['accuracy'])
ax1.set_title('model accuracy')
ax1.set_ylabel('accuracy')
ax1.set_xlabel('epoch')
ax1.legend(['Conventional', 'Dopaminergic'], loc='lower right')

ax3.plot(history.history['loss'])
ax3.plot(dop_history.history['loss'])
ax3.set_title('model loss')
ax3.set_ylabel('loss')
ax3.set_xlabel('epoch')
ax3.legend(['Conventional', 'Dopaminergic'], loc='upper right')

pyplot.show()

		
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
