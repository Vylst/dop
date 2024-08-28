import tensorflow as tf
import numpy as np
import random
import pickle

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from matplotlib import pyplot

from sklearn import datasets
from sklearn.model_selection import train_test_split

def unison_shuffled_copies(a, b):
	assert len(a) == len(b)
	p = np.random.permutation(len(a))
	return a[p], b[p]


# Model / data parameters
num_classes = 10
input_shape = (32, 32, 3)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)

# Scale images to the [0, 1] range
x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255


# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


idx = np.concatenate((np.where(y_train == 0)[0], np.where(y_train == 1)[0]))
x_train_01 = x_train[idx]
y_train_01 = y_train[idx]
idx = np.concatenate((np.where(y_train == 2)[0], np.where(y_train == 3)[0]))
x_train_23 = x_train[idx]
y_train_23 = y_train[idx]
idx = np.concatenate((np.where(y_train == 4)[0], np.where(y_train == 5)[0]))
x_train_45 = x_train[idx]
y_train_45 = y_train[idx]
idx = np.concatenate((np.where(y_train == 6)[0], np.where(y_train == 7)[0]))
x_train_67 = x_train[idx]
y_train_67 = y_train[idx]
idx = np.concatenate((np.where(y_train == 8)[0], np.where(y_train == 9)[0]))
x_train_89 = x_train[idx]
y_train_89 = y_train[idx]

# convert class vectors to binary class matrices
y_train_01 = tf.keras.utils.to_categorical(y_train_01, num_classes)
y_train_23 = tf.keras.utils.to_categorical(y_train_23, num_classes)
y_train_45 = tf.keras.utils.to_categorical(y_train_45, num_classes)
y_train_67 = tf.keras.utils.to_categorical(y_train_67, num_classes)
y_train_89 = tf.keras.utils.to_categorical(y_train_89, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)






class MyCallback(Callback):
	def __init__(self, epoch_variable):
		self.epoch_variable = epoch_variable
		
	def on_epoch_begin(self, epoch, logs=None):
		self.epoch_variable += 1

class DopDense(Layer):

	def __init__(self, activation='softmax', units=32, n_dop=None, threshold=0, refractory_period=0, gamma=0.3):
		'''Initializes the instance attributes'''
		super(DopDense, self).__init__()
		self.units = units
		self.threshold = threshold
		self.ref_period = refractory_period
		self.activation = activation
		self.gamma = gamma
		
		if(n_dop == None):
			self.n_dop = np.random.randint(1, units/2)
		
		else:
			if(n_dop <= units/2):			
				self.n_dop = n_dop	
			else:
				raise ValueError("Dopaminergic neurons must not exceed half of layer size.")
		
		## Generate evenly spaced (linspace) indices for dopaminergic neurons
		self.dop_indices = np.linspace(1, self.units-1, self.n_dop, dtype=np.int32)
		self.dop_neurons = np.zeros(self.units)
		for i in self.dop_indices:
			self.dop_neurons[i] = 1
		self.non_dop_neurons = np.ones(self.units) - self.dop_neurons
		self.non_dop_indices = np.where(self.non_dop_neurons == 1)[0]


		## Placeholders for dopaminergic neuron weights
		self.dop_weights_new = tf.Variable(0., trainable=False)
		self.dop_weights_old = tf.Variable(0., trainable=False)

		## Initialize refraction period indicator for each dopaminergic neuron
		self.indicator = -self.ref_period*np.ones((self.n_dop))
		
		
		
		
		self.impact_array = np.zeros((int(np.sum(self.non_dop_neurons)), self.units))

		for i, non_special_index in enumerate(self.non_dop_indices):
			row = np.zeros(self.units)
			for j, special_index in enumerate(self.dop_indices):
				distance = np.abs(non_special_index - special_index)
				impact = 1 / (2 ** distance)
				row[special_index] = impact
			self.impact_array[i] = row

		print("AAAAAAAAAAAAAA: ", self.impact_array)
		print("AAAAAAAAAAAAAA: ", np.shape(self.impact_array))
		


	def build(self, input_shape):
		'''Create the state of the layer (weights)'''        

		self.batch_ctr = 0#np.array(0, dtype=np.int32)

		self.cntr = np.zeros((150), dtype=np.int32)

		## Number of connections per neuron in this layer, with neurons from the previous layer
		self.n_inputs = input_shape[-1]

		## initialize the weights
		w_init = tf.random_normal_initializer()
		self.w = tf.Variable(name="kernel", initial_value=w_init(shape=(input_shape[-1], self.units), dtype='float32'), trainable=True)

		## initialize the biases
		b_init = tf.zeros_initializer()
		self.b = tf.Variable(name="bias", initial_value=b_init(shape=(self.units,), dtype='float32'), trainable=True)
		
		## Dopaminergic neuron filter
		f = np.zeros((input_shape[-1], self.units), dtype=np.float32)
		for i in range(self.n_dop):
			f[:,self.dop_indices[i]] = 1
		self.filter = tf.Variable(f, dtype=tf.float32, trainable=False)
		
		
		## Instantiate variable where trace will be accumulated
		self.trace = tf.Variable(np.zeros((self.units), dtype=np.float32), trainable=False)
		
		tf.print("input shape", input_shape[-1])
		tf.print("Weight shape: ", np.shape(self.w))
		tf.print("Filter shape: ", np.shape(self.filter))
		


	## Check if training or testing
	def train_test(self):
		return tf.keras.backend.in_train_phase(1, 0)
	

	def call(self, inputs):
		'''Defines the computation from inputs to outputs'''

		#print("\nlearning phase: ", self.train_test())
		
		## Array of D1 D2 gate coefficients for conventional neurons - Left: D1, Right: D2 for each conventional neuron
		tops = np.random.uniform(low=0, high=0.001, size=(self.units - self.n_dop,2))
		bottoms = np.random.uniform(low=0, high=0.001, size=(self.units - self.n_dop,2))
		

		#DO THIS ONLY IN THE TRAINING SECTION TO ACCELERATE LEARNING
		if(self.train_test()):
			## Gather weights from dopaminergic neurons by multiplying with {0,1} filter where 1 indicates dopaminergic neuron
			self.dop_weights_new = tf.math.multiply(self.w, self.filter)
			
			
			#tf.print("\n\n\nepoch: ", epoch_variable, "batch: ", self.batch_ctr)
	


			## There may be an issue where, in the "first batch", the variables are instantiated and no calculations are done, so better start in the "second"
			if(self.batch_ctr > 1):
				
				## Compute differences in dopaminergic weights between batch t and t-1, and average by number of dopaminergic connections per dopaminergic neuron
				dop_neuron_mean_difference = tf.math.reduce_sum( tf.math.subtract(self.dop_weights_new, self.dop_weights_old), axis=0) / self.n_inputs
				

				self.trace = self.gamma*dop_neuron_mean_difference + (1-self.gamma)*self.trace
				

				impact = np.sum(self.trace * self.impact_array, axis = 1)
				
			
				
				
				for i in range(self.units - self.n_dop):
					non_dop_neuron_idx = self.non_dop_indices[i]
					
					if(self.trace[i] > self.threshold):
					
						if(self.trace[i] > 0):
						
							impact[i] += (tops[i][0] - bottoms[i][0])
						else:
							 impact[i] += (-tops[i][1] + bottoms[i][1])
					
						self.w[:, non_dop_neuron_idx].assign(self.w[:, non_dop_neuron_idx] + impact[i])


			
				
			
			## Store current weights as old weights for next batch
			self.dop_weights_old = self.dop_weights_new
			
			## Increment batch counter
			self.batch_ctr = self.batch_ctr + 1
			



			
		#CALCULATE OUTPUT
		output = tf.matmul(inputs, self.w) + self.b
		
		#tf.print("\nunits: ", self.units)
		#tf.print("input: ", inputs.shape)
		#tf.print("weights: ", self.w.shape)
		#tf.print("output: ", output.shape)
		
		
		## Usual linear combination of fully connected layers
		if(self.activation == "relu"):
			return tf.keras.activations.relu(output)
		elif(self.activation == "softmax"):
			return tf.keras.activations.softmax(output)
		else:
			return output



opt = tf.keras.optimizers.Adam(learning_rate=0.001) #0.001 is default for adam
			

n_dop_neurons = 0.5


epoch_variable = np.array(0, dtype=np.int32)
inp = Input(input_shape)
h1 = Conv2D(32, kernel_size=(3, 3), activation="relu")(inp)
h2 = MaxPooling2D(pool_size=(2, 2))(h1)
h3 = Conv2D(64, kernel_size=(3, 3), activation="relu")(h2)
h4 = MaxPooling2D(pool_size=(2, 2))(h3)
h5 = Flatten()(h4)
h6 = Dropout(0.5)(h5)
#h7 = DopDense("relu", 100, int(n_dop_neurons*100), threshold = 0.00018, refractory_period = 2, gamma = 0.3)(h6)
#out = Dense(num_classes, activation=tf.keras.activations.softmax)(h7)
out = DopDense("softmax", num_classes, int(n_dop_neurons*num_classes), threshold = 0.00018, refractory_period = 2, gamma = 0.3)(h6)
dop_model = Model(inputs=inp, outputs=out)
dop_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

dop_model.summary()


#shuffle
x_train, y_train = unison_shuffled_copies(x_train_01, y_train_01)
dop_history_01 = dop_model.fit(x_train, y_train, epochs=20, batch_size=64, verbose=1, callbacks=[MyCallback(epoch_variable)], validation_split=0.2)

#concat and shuffle
x_train_0123 = np.concatenate((x_train_01, x_train_23))
y_train_0123 = np.concatenate((y_train_01, y_train_23))
x_train, y_train = unison_shuffled_copies(x_train_0123, y_train_0123)
dop_history_0123 = dop_model.fit(x_train, y_train, epochs=20, batch_size=64, verbose=1, callbacks=[MyCallback(epoch_variable)], validation_split=0.2)

#concat and shuffle
x_train_012345 = np.concatenate((x_train_0123, x_train_45))
y_train_012345 = np.concatenate((y_train_0123, y_train_45))
x_train, y_train = unison_shuffled_copies(x_train_012345, y_train_012345)
dop_history_012345 = dop_model.fit(x_train, y_train, epochs=20, batch_size=64, verbose=1, callbacks=[MyCallback(epoch_variable)], validation_split=0.2)

#concat and shuffle
x_train_01234567 = np.concatenate((x_train_012345, x_train_67))
y_train_01234567 = np.concatenate((y_train_012345, y_train_67))
x_train, y_train = unison_shuffled_copies(x_train_01234567, y_train_01234567)
dop_history_01234567 = dop_model.fit(x_train, y_train, epochs=20, batch_size=64, verbose=1, callbacks=[MyCallback(epoch_variable)], validation_split=0.2)

#concat and shuffle
x_train_0123456789 = np.concatenate((x_train_01234567, x_train_89))
y_train_0123456789 = np.concatenate((y_train_01234567, y_train_89))
x_train, y_train = unison_shuffled_copies(x_train_0123456789, y_train_0123456789)
dop_history_0123456789 = dop_model.fit(x_train, y_train, epochs=20, batch_size=64, verbose=1, callbacks=[MyCallback(epoch_variable)], validation_split=0.2)


dop_history_accuracy = np.concatenate((dop_history_01.history['accuracy'], dop_history_0123.history['accuracy'], dop_history_012345.history['accuracy'], dop_history_01234567.history['accuracy'], dop_history_0123456789.history['accuracy']))
dop_history_loss = np.concatenate((dop_history_01.history['loss'], dop_history_0123.history['loss'], dop_history_012345.history['loss'], dop_history_01234567.history['loss'], dop_history_0123456789.history['loss']))

dop_history_valaccuracy = np.concatenate((dop_history_01.history['val_accuracy'], dop_history_0123.history['val_accuracy'], dop_history_012345.history['val_accuracy'], dop_history_01234567.history['val_accuracy'], dop_history_0123456789.history['val_accuracy']))
dop_history_valloss = np.concatenate((dop_history_01.history['val_loss'], dop_history_0123.history['val_loss'], dop_history_012345.history['val_loss'], dop_history_01234567.history['val_loss'], dop_history_0123456789.history['val_loss']))





dop_results = dop_model.evaluate(x_test, y_test, batch_size=128)
print("Dopaminergic test loss: ", dop_results[0])
print("Dopaminergic test acc: ", dop_results[1])


with open("SEC_dop_history_d1d2_" + str(n_dop_neurons) + "_dop_neurons.pkl", 'wb') as fp:
	pickle.dump([dop_history_accuracy, dop_history_loss, dop_history_valaccuracy, dop_history_valloss], fp)





'''
fig, ((ax1, ax2), (ax3, ax4)) = pyplot.subplots(2, 2)

ax1.plot(dop_history_accuracy)
ax1.set_title('model accuracy')
ax1.set_ylabel('accuracy')
ax1.set_xlabel('epoch')
ax1.legend(['Conventional', 'Dopaminergic'], loc='lower right')

ax2.plot(dop_history_valaccuracy)
ax2.set_title('validation accuracy')
ax2.set_ylabel('accuracy')
ax2.set_xlabel('epoch')
ax2.legend(['Conventional', 'Dopaminergic'], loc='lower right')

ax3.plot(dop_history_loss)
ax3.set_title('model loss')
ax3.set_ylabel('loss')
ax3.set_xlabel('epoch')
ax3.legend(['Conventional', 'Dopaminergic'], loc='upper right')

ax4.plot(dop_history_valloss)
ax4.set_title('validation loss')
ax4.set_ylabel('loss')
ax4.set_xlabel('epoch')
ax4.legend(['Convetional', 'Dopaminergic'], loc='upper right')


pyplot.show()

'''	
			
			
			
			
			
			
			
			
			
			
			
			
			
