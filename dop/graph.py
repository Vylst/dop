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


fig, (ax1, ax2) = pyplot.subplots(1, 2)


with open("conventional_history.pkl", 'rb') as fp:
		conventional = pickle.load(fp)

list_hist = []
for i in range(5):
	with open("last_layer_10units/dop_history_" + str(i+1) + "_dop_neurons.pkl", 'rb') as fp:
		list_hist.append(pickle.load(fp))
		
	ax1.plot(list_hist[i]['accuracy'])
	ax2.plot(list_hist[i]['loss'])


ax1.plot(conventional['accuracy'])
ax1.set_title('model accuracies')
ax1.set_ylabel('accuracy')
ax1.set_xlabel('epoch')
ax1.legend(['1 Dop', '2 Dop', '3 Dop', '4 Dop', '5 Dop', 'Conventional'], loc='lower right')

ax2.plot(conventional['loss'])
ax2.set_title('model losses')
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
ax2.legend(['1 Dop', '2 Dop', '3 Dop', '4 Dop', '5 Dop', 'Conventional'], loc='lower right')

pyplot.show()

























