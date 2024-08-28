import tensorflow as tf
import numpy as np
import random
import pickle
import time




from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from matplotlib import pyplot
from sklearn import datasets
from sklearn.model_selection import train_test_split
from DopDense_v2 import DopDense, MyCallback
#from DopVarDense import DopVarDense, MyCallback
#from DopRecDense import DopRecDense, MyCallback


# Model / data parameters
num_classes = 10
input_shape = (32, 32, 3)
epochs = 150
batch_size = 64
learning_rate = 0.001
validation_split = 0.2

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

x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.shuffle(buffer_size=1024).batch(batch_size)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)


'''
i = Input(input_shape)
h_1 = Conv2D(32, kernel_size=(3, 3), activation="relu")(i)
h_2 = MaxPooling2D(pool_size=(2, 2))(h_1)
h_3 = Conv2D(64, kernel_size=(3, 3), activation="relu")(h_2)
h_4 = MaxPooling2D(pool_size=(2, 2))(h_3)
h_5 = Flatten()(h_4)
h_6 = Dropout(0.5)(h_5)
o = Dense(num_classes, activation="softmax")(h_6)
model = Model(inputs=i, outputs=o)

'''

'''
inp = Input(input_shape)
conv1 = Conv2D(32, kernel_size=(3, 3), activation="relu")
pool = MaxPooling2D(pool_size=(2, 2))
conv2 = Conv2D(64, kernel_size=(3, 3), activation="relu")
flatten = Flatten()
dropout = Dropout(0.5)
dense = Dense(512, activation="relu")
d = DopRecDense("relu", 512, learning_rate*0.05, learning_rate, dense)
dop = DopRecDense("softmax", 10, learning_rate*0.05, learning_rate, dense)

x = conv1(inp)
x = pool(x)
x = conv2(x)
x = pool(x)
x = flatten(x)
x = dropout(x)
#x = dense(x)
x = d(x)
x = dop(x)
'''

inp = Input(input_shape)
h1 = Conv2D(32, kernel_size=(3, 3), activation="relu")(inp)
h2 = MaxPooling2D(pool_size=(2, 2))(h1)
h3 = Conv2D(64, kernel_size=(3, 3), activation="relu")(h2)
h4 = MaxPooling2D(pool_size=(2, 2))(h3)
h5 = Flatten()(h4)
h6 = Dropout(0.5)(h5)
#out = DopRecDense("softmax", 10, learning_rate*0.05, learning_rate)(h6)
#out = DopVarDense("softmax", 10, learning_rate*0.05, 0)(h6)
#h = DopDense("relu", 512, 100, learning_rate*0.2, 8)(h6)
out = DopDense("softmax", 10, 5, learning_rate*0.2, 8)(h6)
dop_model = Model(inputs=inp, outputs=out)

dop_model.summary()



optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = tf.keras.losses.CategoricalCrossentropy()

dop_train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
dop_train_loss_metric = tf.keras.metrics.Mean()

train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
train_loss_metric = tf.keras.metrics.Mean()

dop_val_acc_metric = tf.keras.metrics.CategoricalAccuracy()
dop_val_loss_metric = tf.keras.metrics.Mean()

val_acc_metric = tf.keras.metrics.CategoricalAccuracy()
val_loss_metric = tf.keras.metrics.Mean()

callback = MyCallback()

acc_list = []
loss_list = []
dop_acc_list = []
dop_loss_list = []
val_acc_list = []
val_loss_list = []
dop_val_acc_list = []
dop_val_loss_list = []


print(np.shape(list(enumerate(train_dataset))))
print(np.shape(list(enumerate(val_dataset))))


for epoch in range(epochs):
	
	start_time = time.time()
	
	callback.on_epoch_begin(epoch)
	
	# Iterate over the batches of the dataset.
	for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

		with tf.GradientTape() as dop_tape:

			dop_logits = dop_model(x_batch_train, training=True)
			loss = loss_fn(y_batch_train, dop_logits)

		
		dop_grads = dop_tape.gradient(loss, dop_model.trainable_weights)
		optimizer.apply_gradients(zip(dop_grads, dop_model.trainable_weights))
		

		
		dop_train_acc_metric.update_state(y_batch_train, dop_logits)
		dop_train_loss_metric.update_state(loss)

	dop_train_acc = dop_train_acc_metric.result()	
	dop_acc_list.append(float(dop_train_acc))
	dop_train_acc_metric.reset_states()
	
	dop_train_loss = dop_train_loss_metric.result()
	dop_loss_list.append(float(dop_train_loss))
	dop_train_loss_metric.reset_states()
	

	# Run a validation loop at the end of each epoch.
	for x_batch_val, y_batch_val in val_dataset:
		#with tf.GradientTape() as dop_val_tape:
		dop_val_logits = dop_model(x_batch_val, training=False)
		loss = loss_fn(y_batch_val, dop_val_logits)
		
		dop_val_acc_metric.update_state(y_batch_val, dop_val_logits)
		dop_val_loss_metric.update_state(loss)

	
	dop_val_acc = dop_val_acc_metric.result()
	dop_val_acc_list.append(float(dop_val_acc))
	dop_val_acc_metric.reset_states()
	
	dop_val_loss = dop_val_loss_metric.result()
	dop_val_loss_list.append(float(dop_val_loss))
	dop_val_loss_metric.reset_states()
	
	print("DOPAMINERGIC")
	print( "Epoch %d - %.2fs - loss: %.5f - accuracy: %.5f - val_loss: %.5f - val_accuracy: %.5f\n" % 
		(epoch,time.time() - start_time,float(dop_train_loss),float(dop_train_acc),float(dop_val_loss),float(dop_val_acc)) )
		
callback.on_training_end()
	
'''
for epoch in range(epochs):
	
	start_time = time.time()
	
	for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
		with tf.GradientTape() as tape:

			logits = model(x_batch_train, training=True)
			loss = loss_fn(y_batch_train, logits)

		grads = tape.gradient(loss, model.trainable_weights)
		optimizer.apply_gradients(zip(grads, model.trainable_weights))
		
		train_acc_metric.update_state(y_batch_train, logits)	
		train_loss_metric.update_state(loss)
		
	train_acc = train_acc_metric.result()
	acc_list.append(float(train_acc))
	train_acc_metric.reset_states()
	
	train_loss = train_loss_metric.result()
	loss_list.append(float(train_loss))
	train_loss_metric.reset_states()
	
	
	for x_batch_val, y_batch_val in val_dataset:
		#with tf.GradientTape() as val_tape:
		val_logits = model(x_batch_val, training=False)
		loss = loss_fn(y_batch_val, val_logits)

		val_acc_metric.update_state(y_batch_val, val_logits)
		val_loss_metric.update_state(loss)			
	
	
	val_acc = val_acc_metric.result()
	val_acc_list.append(float(val_acc))
	val_acc_metric.reset_states()
	
	val_loss = val_loss_metric.result()
	val_loss_list.append(float(val_loss))
	val_loss_metric.reset_states()
	
	print("REGULAR")
	print( "Epoch %d - %.2fs - loss: %.5f - accuracy: %.5f - val_loss: %.5f - val_accuracy: %.5f\n" % 
		(epoch,time.time() - start_time,float(train_loss),float(train_acc),float(val_loss),float(val_acc)) )


'''
dop_history = [dop_acc_list, dop_val_acc_list, dop_loss_list, dop_val_loss_list]
with open("dop_history.pkl", 'wb') as fp:
	pickle.dump(dop_history, fp)
'''	
history = [acc_list, val_acc_list, loss_list, val_loss_list]
with open("history.pkl", 'wb') as fp:
	pickle.dump(history, fp)
'''

fig, ((ax1, ax2), (ax3, ax4)) = pyplot.subplots(2, 2)

'''
ax1.plot(acc_list)
ax2.plot(val_acc_list)
ax3.plot(loss_list)
ax4.plot(val_loss_list)
'''

ax1.plot(dop_acc_list)
ax2.plot(dop_val_acc_list)
ax3.plot(dop_loss_list)
ax4.plot(dop_val_loss_list)


ax1.set_title('model accuracy')
ax1.set_ylabel('accuracy')
ax1.set_xlabel('epoch')
ax1.legend(['Conventional', 'Dopaminergic'], loc='lower right')

ax2.set_title('validation accuracy')
ax2.set_ylabel('accuracy')
ax2.set_xlabel('epoch')
ax2.legend(['Conventional', 'Dopaminergic'], loc='lower right')

ax3.set_title('model loss')
ax3.set_ylabel('loss')
ax3.set_xlabel('epoch')
ax3.legend(['Conventional', 'Dopaminergic'], loc='upper right')

ax4.set_title('validation loss')
ax4.set_ylabel('loss')
ax4.set_xlabel('epoch')
ax4.legend(['Convetional', 'Dopaminergic'], loc='upper right')

pyplot.show()


























