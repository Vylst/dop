import tensorflow as tf
import numpy as np
import random
import pickle
import time

tf.config.run_functions_eagerly(True)

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from matplotlib import pyplot
from sklearn import datasets
from sklearn.model_selection import train_test_split
from DopDense import DopDense, MyCallback


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

size = np.shape(y_train)[0]

# Scale images to the [0, 1] range
x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255

x_train_1 = []
y_train_1 = []
x_train_2 = []
y_train_2 = []

for i in range(len(y_train)):
	if(y_train[i] > 4):
		x_train_2.append(x_train[i])
		y_train_2.append(y_train[i])
	else:
		x_train_1.append(x_train[i])
		y_train_1.append(y_train[i])

x_train_1 = np.array(x_train_1)
y_train_1 = np.array(y_train_1)
x_train_2 = np.array(x_train_2)
y_train_2 = np.array(y_train_2)

# Make sure images have shape (28, 28, 1)
x_train_1 = np.expand_dims(x_train_1, -1)
x_train_2 = np.expand_dims(x_train_2, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train_1 shape:", x_train_1.shape)
print(x_train_1.shape[0], "train samples")
print("x_train_2 shape:", x_train_2.shape)
print(x_train_2.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train_1 = tf.keras.utils.to_categorical(y_train_1, num_classes)
y_train_2 = tf.keras.utils.to_categorical(y_train_2, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


print("Shape x_train_1: ", x_train_1.shape)
print("Shape y_train_1: ", y_train_1.shape)
print("Shape x_train_2: ", x_train_2.shape)
print("Shape y_train_2: ", y_train_2.shape)


x_val_1 = x_train_1[-int(size*validation_split/2):]
y_val_1 = y_train_1[-int(size*validation_split/2):]
x_train_1 = x_train_1[:-int(size*validation_split/2)]
y_train_1 = y_train_1[:-int(size*validation_split/2)]
x_val_2 = x_train_2[-int(size*validation_split/2):]
y_val_2 = y_train_2[-int(size*validation_split/2):]
x_train_2 = x_train_2[:-int(size*validation_split/2)]
y_train_2 = y_train_2[:-int(size*validation_split/2)]

val_dataset_1 = tf.data.Dataset.from_tensor_slices((x_val_1, y_val_1))
val_dataset_1 = val_dataset_1.shuffle(buffer_size=1024).batch(batch_size)
val_dataset_2 = tf.data.Dataset.from_tensor_slices((x_val_2, y_val_2))
val_dataset_2 = val_dataset_2.shuffle(buffer_size=1024).batch(batch_size)

train_dataset_1 = tf.data.Dataset.from_tensor_slices((x_train_1, y_train_1))
train_dataset_1 = train_dataset_1.shuffle(buffer_size=1024).batch(batch_size)
train_dataset_2 = tf.data.Dataset.from_tensor_slices((x_train_2, y_train_2))
train_dataset_2 = train_dataset_2.shuffle(buffer_size=1024).batch(batch_size)

i = Input(input_shape)
h_1 = Conv2D(32, kernel_size=(3, 3), activation="relu")(i)
h_2 = MaxPooling2D(pool_size=(2, 2))(h_1)
h_3 = Conv2D(64, kernel_size=(3, 3), activation="relu")(h_2)
h_4 = MaxPooling2D(pool_size=(2, 2))(h_3)
h_5 = Flatten()(h_4)
h_6 = Dropout(0.5)(h_5)
o = Dense(num_classes, activation="softmax")(h_6)
model = Model(inputs=i, outputs=o)

inp = Input(input_shape)
h1 = Conv2D(32, kernel_size=(3, 3), activation="relu")(inp)
h2 = MaxPooling2D(pool_size=(2, 2))(h1)
h3 = Conv2D(64, kernel_size=(3, 3), activation="relu")(h2)
h4 = MaxPooling2D(pool_size=(2, 2))(h3)
h5 = Flatten()(h4)
h6 = Dropout(0.5)(h5)
out = DopDense("softmax", num_classes, 5, 0.0002, 8)(h6)
dop_model = Model(inputs=inp, outputs=out)


optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

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


print(np.shape(list(enumerate(train_dataset_1))))
print(np.shape(list(enumerate(val_dataset_1))))
print(np.shape(list(enumerate(train_dataset_2))))
print(np.shape(list(enumerate(val_dataset_2))))


for epoch in range(epochs):
	
	start_time = time.time()
	
	callback.on_epoch_begin(epoch)
	
	
	if(epoch < epochs/2):
	
		# Iterate over the batches of the dataset.
		for step, (x_batch_train, y_batch_train) in enumerate(train_dataset_1):

			with tf.GradientTape() as dop_tape:

				dop_logits = dop_model(x_batch_train, training=True)
				loss1 = loss_fn(y_batch_train, dop_logits)

			dop_grads = dop_tape.gradient(loss1, dop_model.trainable_weights)
			optimizer.apply_gradients(zip(dop_grads, dop_model.trainable_weights))
			
			dop_train_acc_metric.update_state(y_batch_train, dop_logits)
			dop_train_loss_metric.update_state(loss1)


		# Run a validation loop at the end of each epoch.
		for x_batch_val, y_batch_val in val_dataset_1:

			dop_val_logits = dop_model(x_batch_val, training=False)
			loss2 = loss_fn(y_batch_val, dop_val_logits)
			
			dop_val_acc_metric.update_state(y_batch_val, dop_val_logits)
			dop_val_loss_metric.update_state(loss2)

	else:
		for step, (x_batch_train, y_batch_train) in enumerate(train_dataset_2):

			with tf.GradientTape() as dop_tape:

				dop_logits = dop_model(x_batch_train, training=True)
				loss1 = loss_fn(y_batch_train, dop_logits)

			dop_grads = dop_tape.gradient(loss1, dop_model.trainable_weights)
			optimizer.apply_gradients(zip(dop_grads, dop_model.trainable_weights))
			
			dop_train_acc_metric.update_state(y_batch_train, dop_logits)
			dop_train_loss_metric.update_state(loss1)

		for x_batch_val, y_batch_val in val_dataset_2:

			dop_val_logits = dop_model(x_batch_val, training=False)
			loss2 = loss_fn(y_batch_val, dop_val_logits)
			
			dop_val_acc_metric.update_state(y_batch_val, dop_val_logits)
			dop_val_loss_metric.update_state(loss2)
	
	dop_train_acc = dop_train_acc_metric.result()	
	dop_acc_list.append(float(dop_train_acc))
	dop_train_acc_metric.reset_states()
	
	dop_train_loss = dop_train_loss_metric.result()
	dop_loss_list.append(float(dop_train_loss))
	dop_train_loss_metric.reset_states()
	
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
	

for epoch in range(epochs):
	
	start_time = time.time()
	
	if(epoch < epochs/2):
	
		for step, (x_batch_train, y_batch_train) in enumerate(train_dataset_1):
			with tf.GradientTape() as tape:

				logits = model(x_batch_train, training=True)
				loss1 = loss_fn(y_batch_train, logits)

			grads = tape.gradient(loss1, model.trainable_weights)
			optimizer.apply_gradients(zip(grads, model.trainable_weights))
			
			train_acc_metric.update_state(y_batch_train, logits)	
			train_loss_metric.update_state(loss1)
			
		
		
		
		for x_batch_val, y_batch_val in val_dataset_1:

			val_logits = model(x_batch_val, training=False)
			loss2 = loss_fn(y_batch_val, val_logits)

			val_acc_metric.update_state(y_batch_val, val_logits)
			val_loss_metric.update_state(loss2)		
			
	else:
		for step, (x_batch_train, y_batch_train) in enumerate(train_dataset_2):
			with tf.GradientTape() as tape:

				logits = model(x_batch_train, training=True)
				loss1 = loss_fn(y_batch_train, logits)

			grads = tape.gradient(loss1, model.trainable_weights)
			optimizer.apply_gradients(zip(grads, model.trainable_weights))
			
			train_acc_metric.update_state(y_batch_train, logits)	
			train_loss_metric.update_state(loss1)
			
		
		
		
		for x_batch_val, y_batch_val in val_dataset_2:

			val_logits = model(x_batch_val, training=False)
			loss2 = loss_fn(y_batch_val, val_logits)

			val_acc_metric.update_state(y_batch_val, val_logits)
			val_loss_metric.update_state(loss2)		
	
	train_acc = train_acc_metric.result()
	acc_list.append(float(train_acc))
	train_acc_metric.reset_states()
	
	train_loss = train_loss_metric.result()
	loss_list.append(float(train_loss))
	train_loss_metric.reset_states()
	
	val_acc = val_acc_metric.result()
	val_acc_list.append(float(val_acc))
	val_acc_metric.reset_states()
	
	val_loss = val_loss_metric.result()
	val_loss_list.append(float(val_loss))
	val_loss_metric.reset_states()
	
	print("REGULAR")
	print( "Epoch %d - %.2fs - loss: %.5f - accuracy: %.5f - val_loss: %.5f - val_accuracy: %.5f\n" % 
		(epoch,time.time() - start_time,float(train_loss),float(train_acc),float(val_loss),float(val_acc)) )
	


fig, ((ax1, ax2), (ax3, ax4)) = pyplot.subplots(2, 2)

ax1.plot(acc_list)
ax1.plot(dop_acc_list)
ax1.set_title('model accuracy')
ax1.set_ylabel('accuracy')
ax1.set_xlabel('epoch')
ax1.legend(['Conventional', 'Dopaminergic'], loc='lower right')

ax2.plot(val_acc_list)
ax2.plot(dop_val_acc_list)
ax2.set_title('validation accuracy')
ax2.set_ylabel('accuracy')
ax2.set_xlabel('epoch')
ax2.legend(['Conventional', 'Dopaminergic'], loc='lower right')

ax3.plot(loss_list)
ax3.plot(dop_loss_list)
ax3.set_title('model loss')
ax3.set_ylabel('loss')
ax3.set_xlabel('epoch')
ax3.legend(['Conventional', 'Dopaminergic'], loc='upper right')

ax4.plot(val_loss_list)
ax4.plot(dop_val_loss_list)
ax4.set_title('validation loss')
ax4.set_ylabel('loss')
ax4.set_xlabel('epoch')
ax4.legend(['Convetional', 'Dopaminergic'], loc='upper right')

pyplot.show()













