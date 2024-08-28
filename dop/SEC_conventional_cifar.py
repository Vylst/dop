import tensorflow as tf
import numpy as np
import random
import pickle


#tf.config.run_functions_eagerly(True)
#tf.data.experimental.enable_debug_mode()

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model, Sequential
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



opt = tf.keras.optimizers.Adam(learning_rate=0.001) #0.001 is default for adam
			
i = Input(input_shape)
h_1 = Conv2D(32, kernel_size=(3, 3), activation="relu")(i)
h_2 = MaxPooling2D(pool_size=(2, 2))(h_1)
h_3 = Conv2D(64, kernel_size=(3, 3), activation="relu")(h_2)
h_4 = MaxPooling2D(pool_size=(2, 2))(h_3)
h_5 = Flatten()(h_4)
h_6 = Dropout(0.5)(h_5)
o = Dense(num_classes, activation=tf.keras.activations.softmax)(h_6)
model = Model(inputs=i, outputs=o)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
#history = model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=1, validation_split=0.2)


#shuffle
x_train, y_train = unison_shuffled_copies(x_train_01, y_train_01)
history_01 = model.fit(x_train, y_train, epochs=20, batch_size=64, verbose=1, validation_split=0.2)

#concat and shuffle
x_train_0123 = np.concatenate((x_train_01, x_train_23))
y_train_0123 = np.concatenate((y_train_01, y_train_23))
x_train, y_train = unison_shuffled_copies(x_train_0123, y_train_0123)
history_0123 = model.fit(x_train, y_train, epochs=20, batch_size=64, verbose=1, validation_split=0.2)

#concat and shuffle
x_train_012345 = np.concatenate((x_train_0123, x_train_45))
y_train_012345 = np.concatenate((y_train_0123, y_train_45))
x_train, y_train = unison_shuffled_copies(x_train_012345, y_train_012345)
history_012345 = model.fit(x_train, y_train, epochs=20, batch_size=64, verbose=1, validation_split=0.2)

#concat and shuffle
x_train_01234567 = np.concatenate((x_train_012345, x_train_67))
y_train_01234567 = np.concatenate((y_train_012345, y_train_67))
x_train, y_train = unison_shuffled_copies(x_train_01234567, y_train_01234567)
history_01234567 = model.fit(x_train, y_train, epochs=20, batch_size=64, verbose=1, validation_split=0.2)

#concat and shuffle
x_train_0123456789 = np.concatenate((x_train_01234567, x_train_89))
y_train_0123456789 = np.concatenate((y_train_01234567, y_train_89))
x_train, y_train = unison_shuffled_copies(x_train_0123456789, y_train_0123456789)
history_0123456789 = model.fit(x_train, y_train, epochs=20, batch_size=64, verbose=1, validation_split=0.2)


history_accuracy = np.concatenate((history_01.history['accuracy'], history_0123.history['accuracy'], history_012345.history['accuracy'], history_01234567.history['accuracy'], history_0123456789.history['accuracy']))
history_loss = np.concatenate((history_01.history['loss'], history_0123.history['loss'], history_012345.history['loss'], history_01234567.history['loss'], history_0123456789.history['loss']))

history_valaccuracy = np.concatenate((history_01.history['val_accuracy'], history_0123.history['val_accuracy'], history_012345.history['val_accuracy'], history_01234567.history['val_accuracy'], history_0123456789.history['val_accuracy']))
history_valloss = np.concatenate((history_01.history['val_loss'], history_0123.history['val_loss'], history_012345.history['val_loss'], history_01234567.history['val_loss'], history_0123456789.history['val_loss']))




results = model.evaluate(x_test, y_test, batch_size=128)
print("RESULTS")
print(np.shape(results))

print("Conventional test loss: ", results[0])
print("Conventional test acc: ", results[1])


with open("SEC_conventional_history.pkl", 'wb') as fp:
	pickle.dump([history_accuracy, history_loss, history_valaccuracy, history_valloss], fp)
	
	



fig, ((ax1, ax2), (ax3, ax4)) = pyplot.subplots(2, 2)

ax1.plot(history_accuracy)
ax1.set_title('model accuracy')
ax1.set_ylabel('accuracy')
ax1.set_xlabel('epoch')
ax1.legend(['Conventional', 'Dopaminergic'], loc='lower right')


ax2.plot(history_valaccuracy)
ax2.set_title('validation accuracy')
ax2.set_ylabel('accuracy')
ax2.set_xlabel('epoch')
ax2.legend(['Conventional', 'Dopaminergic'], loc='lower right')


ax3.plot(history_loss)
ax3.set_title('model loss')
ax3.set_ylabel('loss')
ax3.set_xlabel('epoch')
ax3.legend(['Conventional', 'Dopaminergic'], loc='upper right')


ax4.plot(history_valloss)
ax4.set_title('validation loss')
ax4.set_ylabel('loss')
ax4.set_xlabel('epoch')
ax4.legend(['Convetional', 'Dopaminergic'], loc='upper right')


pyplot.show()












