import tensorflow as tf
import numpy as np
import random
import pickle

import seaborn as sns

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from matplotlib import pyplot

from sklearn import datasets
from sklearn.model_selection import train_test_split

sns.set()

fig, (ax1, ax2) = pyplot.subplots(1, 2)


with open("conventional_history.pkl", 'rb') as fp:
		conventional = pickle.load(fp)

'''
with open("dop_history_0.1_dop_neurons.pkl", 'rb') as fp:
		dop_history_1_dop_neuron = pickle.load(fp)
with open("dop_history_0.2_dop_neurons.pkl", 'rb') as fp:
		dop_history_2_dop_neuron = pickle.load(fp)
with open("dop_history_0.3_dop_neurons.pkl", 'rb') as fp:
		dop_history_3_dop_neuron = pickle.load(fp)
with open("dop_history_0.4_dop_neurons.pkl", 'rb') as fp:
		dop_history_4_dop_neuron = pickle.load(fp)
with open("dop_history_0.5_dop_neurons.pkl", 'rb') as fp:
		dop_history_5_dop_neuron = pickle.load(fp)
		
with open("dop_history_d1d2_0.1_dop_neurons.pkl", 'rb') as fp:
		dop_history_d1d2_1_dop_neuron = pickle.load(fp)
with open("dop_history_d1d2_0.2_dop_neurons.pkl", 'rb') as fp:
		dop_history_d1d2_2_dop_neuron = pickle.load(fp)
with open("dop_history_d1d2_0.3_dop_neurons.pkl", 'rb') as fp:
		dop_history_d1d2_3_dop_neuron = pickle.load(fp)
with open("dop_history_d1d2_0.4_dop_neurons.pkl", 'rb') as fp:
		dop_history_d1d2_4_dop_neuron = pickle.load(fp)
with open("dop_history_d1d2_0.5_dop_neurons.pkl", 'rb') as fp:
		dop_history_d1d2_5_dop_neuron = pickle.load(fp)
'''		




list_hist = []
for i in range(5):
	print(str((i+1)*0.1))
	with open("dop_history_" + str(round((i+1)*0.1,1)) + "_dop_neurons.pkl", 'rb') as fp:
		list_hist.append(pickle.load(fp))
		
	ax1.plot(list_hist[i]['accuracy'])
	ax2.plot(list_hist[i]['loss'])
	
list_hist = []
for i in range(5):
	with open("dop_history_d1d2_" + str(round((i+1)*0.1,1)) + "_dop_neurons.pkl", 'rb') as fp:
		list_hist.append(pickle.load(fp))
		
	ax1.plot(list_hist[i]['accuracy'])
	ax2.plot(list_hist[i]['loss'])


ax1.plot(conventional['accuracy'], color = 'black')
ax1.set_title('model accuracies')
ax1.set_ylabel('accuracy')
ax1.set_xlabel('epoch')
ax1.legend(['U 1 Dop', 'U 2 Dop', 'U 3 Dop', 'U 4 Dop', 'U 5 Dop', 'G 1 Dop', 'G 2 Dop', 'G 3 Dop', 'G 4 Dop', 'G 5 Dop', 'Conventional'], loc='lower right')

ax2.plot(conventional['loss'], color = 'black')
ax2.set_title('model losses')
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
ax2.legend(['U 1 Dop', 'U 2 Dop', 'U 3 Dop', 'U 4 Dop', 'U 5 Dop', 'G 1 Dop', 'G 2 Dop', 'G 3 Dop', 'G 4 Dop', 'G 5 Dop', 'Conventional'], loc='upper right')

pyplot.show()

























