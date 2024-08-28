import numpy as np
import tensorflow as tf
import tensorflow.keras.initializers as initializers

tf.get_logger().setLevel('ERROR')

# Needed for animation
import matplotlib.pyplot as plt

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.constraints import maxnorm

'''------------------------ DATA ------------------------'''
num_classes = 10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)


# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
'''------------------------------------------------------'''

class NN(object):

	def __init__(self, x, y, action_dim, in_shape=(32, 32, 3), num_classes=10):
		'''
		constructor
		:param in_shape: shape of input to network
		:param num_classes: number of classes to learn
		:param action_dim: number of possible actions (possible learning rates)
		:param x,y: Full data
		'''
		

		self.base_model = Sequential()
		(self.base_model).add(Conv2D(32, (3, 3), input_shape=(32,32,3), activation='relu', padding='same')) 
		(self.base_model).add(Dropout(0.2)) 
		(self.base_model).add(Conv2D(32, (3, 3), activation='relu', padding='same')) 
		(self.base_model).add(MaxPooling2D(pool_size=(2, 2))) 
		
		
		self.main_model = Sequential()
		(self.main_model).add(Conv2D(64, (3, 3), activation='relu', padding='same')) 
		(self.main_model).add(Dropout(0.2)) 
		(self.main_model).add(Conv2D(64, (3, 3), activation='relu', padding='same')) 
		(self.main_model).add(MaxPooling2D(pool_size=(2, 2))) 
		(self.main_model).add(Conv2D(128, (3, 3), activation='relu', padding='same')) 
		(self.main_model).add(Dropout(0.2)) 
		(self.main_model).add(Conv2D(128, (3, 3), activation='relu', padding='same')) 
		(self.main_model).add(MaxPooling2D(pool_size=(2, 2))) 
		(self.main_model).add(Flatten()) 
		(self.main_model).add(Dropout(0.2)) 
		(self.main_model).add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3))) 
		(self.main_model).add(Dropout(0.2)) 
		(self.main_model).add(Dense(512, activation='relu', kernel_constraint=maxnorm(3))) 
		(self.main_model).add(Dropout(0.2)) 
		(self.main_model).add(Dense(num_classes, activation='softmax'))
		
		
		(self.base_model).trainable = False
		self.lr_model = Sequential()
		(self.lr_model).add(self.base_model)
		(self.lr_model).add(Flatten())
		(self.lr_model).add(Dense(512, activation='relu', kernel_constraint=maxnorm(3))) 
		(self.lr_model).add(Dropout(0.2)) 
		(self.lr_model).add(Dense(32, activation="relu", kernel_initializer=initializers.he_normal()))
		(self.lr_model).add(Dense(action_dim, activation="linear", kernel_initializer=initializers.Zeros()))
		

		
		self.data = [x, y]
		
	def mean_squared_error_loss(self, q_value: tf.Tensor, reward: tf.Tensor) -> tf.Tensor:
		"""Compute mean squared error loss"""
		q_value = tf.cast(q_value, dtype=tf.float32)
		reward = tf.cast(reward, dtype=tf.float32)
		
		loss = 0.5 * (q_value - reward) ** 2

		#print("mse")

		return loss
		
		

	def forward_pass(self, model, x):
		
		out = model(x)
	
		#print("fp")
	
		return out
		
	def eval(self, x_test, y_test):
	
		(self.main_model).compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
		score = (self.main_model).evaluate(x_test, y_test, verbose=0)
		
		#print("eval")
		
		return score
	
	def full_pass(self, x, y, lr) -> None:
	
		opt = tf.keras.optimizers.Adam(learning_rate=lr)
		(self.main_model).compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
		(self.main_model).train_on_batch(x, y)
		
		#print("bip")
		
		return
		
	def get_reward(self, lr, x_test, y_test) -> tf.Tensor:
		bs = len(y_test)
		opt = tf.keras.optimizers.Adam(learning_rate=lr)
		(self.main_model).compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
		results = (self.main_model).evaluate(x_test, y_test, batch_size=bs)
		reward = tf.constant( [ np.mean(results[1]) ] )
		
		#print("gr")
		
		return reward
		
	def get_sample(self, n_batch=128):
	
		ix = np.random.randint(0, (self.data)[0].shape[0], n_batch)
		x = self.data[0][ix]
		y = self.data[1][ix]
		
		#print("gs")
		
		return x,y
		
		
	def train(self, num_episodes, lrs, exploration_rate=0.1) -> None:
		
		opt = tf.keras.optimizers.Adam(learning_rate=0.01)
		
		for i in range(num_episodes + 1):
			with tf.GradientTape() as tape:
			
				x_test, y_test = self.get_sample(n_batch = 10)
			
				q_values = self.forward_pass(self.lr_model,x_test)


				q_values = tf.math.reduce_mean(q_values, axis=0)
				q_values = tf.reshape(q_values, shape=(1,len(q_values)))

				
				epsilon = np.random.rand()
				if(epsilon <= exploration_rate):
					# Select random learning rate
					action = np.random.choice(len(lrs))
		
				else:
					# Select learning rate with highest q-value
					action = np.argmax(q_values)
					

				
				reward = self.get_reward(lrs[action], x_test, y_test)
				print("LR:", lrs[action], "Reward:", reward)
				q_value = q_values[0, action]

				
				loss_value = self.mean_squared_error_loss(q_value, reward)

				
				grads = tape.gradient(loss_value[0], (self.lr_model).trainable_variables)
				opt.apply_gradients(zip(grads, (self.lr_model).trainable_variables))
				
				
				
				if np.mod(i, 10) == 0:
					print("\n======episode", i, "======")
					print("Q-values", ["%.3f" % n for n in q_values[0]])
					print("Rel. deviation", ["%.3f" % float((q_values[0, i] - lrs[i]) / lrs[i]) for i in range(len(q_values[0])) ],)
				
		
		for i in range(20):
			x, y = self.get_sample()
			self.full_pass(x, y, lrs[action])

		return
		
'''----------------------- MISC -------------------------'''
def plot(q_values: tf.Tensor, bandits: np.array) -> None:
	"""Plot bar chart with selection probability per bandit"""
	q_values_plot = [
		q_values[0],
		q_values[1],
		q_values[2],
		q_values[3],
	]
	bandit_plot = [
		bandits[0],
		bandits[1],
		bandits[2],
		bandits[3],
	]
	width = 0.4
	x = np.arange(len(bandits))
	fig, ax = plt.subplots()
	ax.bar(x - width / 2, q_values_plot, width, label="Q-values")
	ax.bar(x + width / 2, bandit_plot, width, label="True values")

	# Add labels and legend
	ax.set_xticks([0, 1, 2, 3])
	ax.set_xticklabels(["1", "2", "3", "4"])

	plt.xlabel("Bandit")
	plt.ylabel("Value")
	plt.legend(loc="best")

	plt.show()

	return

'''----------------------- EXEC -------------------------'''

learning_rates = np.array([0.01])
exploration_rate = 0.1
num_episodes = 10
n_epochs = 20

nn = NN(x=x_train, y=y_train, action_dim = len(learning_rates))

for i in range(n_epochs):
	nn.train(num_episodes, learning_rates)


score = nn.eval(x_test, y_test)
print("Test loss:", score[0])
print("Test accuracy:", score[1])











