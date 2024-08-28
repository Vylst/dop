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


# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
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



history = model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=1, validation_split=0.2)


results = model.evaluate(x_test, y_test, batch_size=128)
print("RESULTS")
print(np.shape(results))

print("Conventional test loss: ", results[0])
print("Conventional test acc: ", results[1])

with open("conventional_history.pkl", 'wb') as fp:
	pickle.dump(history.history, fp)
	
fig, ((ax1, ax2), (ax3, ax4)) = pyplot.subplots(2, 2)

ax1.plot(history.history['accuracy'])
ax1.set_title('model accuracy')
ax1.set_ylabel('accuracy')
ax1.set_xlabel('epoch')
ax1.legend(['Conventional', 'Dopaminergic'], loc='lower right')


ax2.plot(history.history['val_accuracy'])
ax2.set_title('validation accuracy')
ax2.set_ylabel('accuracy')
ax2.set_xlabel('epoch')
ax2.legend(['Conventional', 'Dopaminergic'], loc='lower right')


ax3.plot(history.history['loss'])
ax3.set_title('model loss')
ax3.set_ylabel('loss')
ax3.set_xlabel('epoch')
ax3.legend(['Conventional', 'Dopaminergic'], loc='upper right')


ax4.plot(history.history['val_loss'])
ax4.set_title('validation loss')
ax4.set_ylabel('loss')
ax4.set_xlabel('epoch')
ax4.legend(['Convetional', 'Dopaminergic'], loc='upper right')


pyplot.show()












