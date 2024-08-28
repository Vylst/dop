import tensorflow as tf
import numpy as np
import random
import pickle
import time

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn import datasets
from sklearn.model_selection import train_test_split
from DopDense_v2 import DopDense, MyCallback
from matplotlib import pyplot

# Model / data parameters
num_classes = 3
input_shape = (4,)
epochs = 5
batch_size = 32
learning_rate = 0.001
validation_split = 0.2

iris = datasets.load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

size = np.shape(y_train)[0]
print("\Training size: ", size, "\n")

x_val = x_train[-int(validation_split*size):]
y_val = y_train[-int(validation_split*size):]
x_train = x_train[:-int(validation_split*size)]
y_train = y_train[:-int(validation_split*size)]
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.shuffle(buffer_size=1024).batch(batch_size)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)



inp = Input((4,))
h1 = DopDense('relu', 5, 2, 0, 0)(inp)
out = Dense(3, activation=tf.keras.activations.softmax)(h1)
model = Model(inputs=inp, outputs=out)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
train_loss_metric = tf.keras.metrics.Mean()

val_acc_metric = tf.keras.metrics.CategoricalAccuracy()
val_loss_metric = tf.keras.metrics.Mean()

acc_list = []
loss_list = []
val_acc_list = []
val_loss_list = []

callback = MyCallback()

for epoch in range(epochs):
	
	start_time = time.time()
	
	callback.on_epoch_begin(epoch)
	
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
	
	print("Dopaminergic")
	print( "Epoch %d - %.2fs - loss: %.5f - accuracy: %.5f - val_loss: %.5f - val_accuracy: %.5f\n" % 
		(epoch,time.time() - start_time,float(train_loss),float(train_acc),float(val_loss),float(val_acc)) )

callback.on_training_end()

fig, ((ax1, ax2), (ax3, ax4)) = pyplot.subplots(2, 2)

ax1.plot(acc_list)
ax1.set_title('model accuracy')
ax1.set_ylabel('accuracy')
ax1.set_xlabel('epoch')
ax1.legend(['Conventional', 'Dopaminergic'], loc='lower right')

ax2.plot(val_acc_list)
ax2.set_title('validation accuracy')
ax2.set_ylabel('accuracy')
ax2.set_xlabel('epoch')
ax2.legend(['Conventional', 'Dopaminergic'], loc='lower right')

ax3.plot(loss_list)
ax3.set_title('model loss')
ax3.set_ylabel('loss')
ax3.set_xlabel('epoch')
ax3.legend(['Conventional', 'Dopaminergic'], loc='upper right')

ax4.plot(val_loss_list)
ax4.set_title('validation loss')
ax4.set_ylabel('loss')
ax4.set_xlabel('epoch')
ax4.legend(['Convetional', 'Dopaminergic'], loc='upper right')

pyplot.show()















