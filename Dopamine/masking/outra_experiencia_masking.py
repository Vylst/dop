import numpy as np

from tensorflow import keras
from tensorflow.keras import layers

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


task = Input(shape=(3,))
m1 = Dense(25, activation='relu', kernel_initializer='he_uniform')(data)
q_out = Dense(1, activation='sigmoid')(m1)
mask_model = Model(inputs=task, outputs=q_out)


inp = keras.Input(shape=input_shape)
h1 = layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(inp)
h2 = layers.MaxPooling2D(pool_size=(2, 2))(h1)
h3 = layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(h2)
h4 = layers.MaxPooling2D(pool_size=(2, 2))(h3)
h5 = layers.Flatten()(h4)
h6 = layers.Dropout(0.5)(h5)
out = layers.Dense(num_classes, activation="softmax")(h6)
model.summary()

batch_size = 128
epochs = 15

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

