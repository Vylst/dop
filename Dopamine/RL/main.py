import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, Dense, Flatten, Conv2D, MaxPooling2D, Dropout

# Model / data parameters
num_classes = 10
input_shape = (32, 32, 3)

i = Input(input_shape)
h_1 = Conv2D(32, kernel_size=(3, 3), activation="relu")(i)
h_2 = MaxPooling2D(pool_size=(2, 2))(h_1)
h_3 = Conv2D(64, kernel_size=(3, 3), activation="relu")(h_2)
h_4 = MaxPooling2D(pool_size=(2, 2))(h_3)
h_5 = Flatten()(h_4)
h_6 = Dropout(0.5)(h_5)
o = Dense(num_classes, activation="softmax", name="dense_1")(h_6)
model = Model(inputs=i, outputs=o)

model.summary()

def get_stimulus(size):

	return np.random.randint(0, size)
	 
	

def perform_action(action, stimulus, label):
	
	stimulus = np.reshape(stimulus, (1,32,32,3))
	label = np.reshape(label, (1,10))

	print("doing action: ", action, "\n")
		
	if(action==0):
		
		model.fit(stimulus, label, epochs=1, verbose=0)
		loss, acc = model.evaluate(stimulus, label, verbose=0)
		reward = tf.keras.activations.sigmoid(-(0.5*loss+4))
		
	elif(action==1):
		model.fit(stimulus, label, epochs=1, verbose=0)
		reward = 0.5
	
	elif(action==2):
		loss, acc = model.evaluate(stimulus, label, verbose=0)
		reward = tf.keras.activations.sigmoid(-(0.5*loss+4))
	else:
		reward = -0.5

	l = dict([(layer.name, layer) for layer in model.layers])
	w = np.reshape(l['dense_1'].get_weights()[0], (1,2304,10))

	return reward, w




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


# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)




q_1 = Input((2304,10))
q_2 = Flatten()(q_1)
q_3 = Dense(512, activation="sigmoid")(q_2)
q_o = Dense(4, activation="linear")(q_3)
q_model = Model(inputs=q_1, outputs=q_o)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
q_model.compile(loss='mse', optimizer='adam', metrics=['mae'])

q_model.summary()






# now execute the q learning
y = 0.95
eps = 0.5
decay_factor = 0.999
r_avg_list = []
cntr = 0
num_episodes = 10

for i in range(num_episodes):

	sti_idx = get_stimulus(x_train.shape[0])
	sti = x_train[sti_idx]
	
	l = dict([(layer.name, layer) for layer in model.layers])
	state = np.reshape(l['dense_1'].get_weights()[0], (1,2304,10))


	
	eps *= decay_factor
	done = False
	r_sum = 0
	
	while not done:
	
	
		if np.random.random() < eps:
			a = np.random.randint(0, 4)
		else:
			
			a = np.argmax(q_model.predict(state))
			
		
		
		reward, new_state = perform_action(a, x_train[sti_idx], y_train[sti_idx])
		
		
		target = reward + y * np.max(q_model.predict(new_state))
		target_vec = q_model.predict(state)[0]
		target_vec[a] = target
		
		
		q_model.fit(state, target_vec.reshape(1, 4), epochs=1, verbose=0)
		state = new_state
		r_sum += reward
		
		if(cntr % 100 == 0):
			[loss, acc] = model.evaluate(x_test, y_test)
			print("Loss: ", loss, "  Acc: ", acc, "\n")
		
		cntr += 1
		
	r_avg_list.append(r_sum / 1000)
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
