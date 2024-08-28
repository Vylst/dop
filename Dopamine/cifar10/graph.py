import pickle
from matplotlib import pyplot
import tensorflow as tf
import numpy as np

'''
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
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


print(x_train_1.shape)
print(y_train_1.shape)
print(x_train_2.shape)
print(y_train_2.shape)

print(y_train_1)
print(y_train_2)
'''




with open("dop_history.pkl", "rb") as fp:
	dh = pickle.load(fp)

with open("baseline.pkl", "rb") as fp:
	h = pickle.load(fp)



ax1 = pyplot.subplot(3, 2, 1)
ax1.plot(h[0])
ax1.plot(dh[0])
ax1.set_title('model accuracy')
ax1.set_ylabel('accuracy')
ax1.set_xlabel('epoch')
ax1.legend(['Conventional', 'Dopaminergic'], loc='lower right')

ax2 = pyplot.subplot(3, 2, 2)
ax2.plot(h[1])
ax2.plot(dh[1])
ax2.set_title('validation accuracy')
ax2.set_ylabel('accuracy')
ax2.set_xlabel('epoch')
ax2.legend(['Conventional', 'Dopaminergic'], loc='lower right')
    
ax3 = pyplot.subplot(3, 2, 3)
ax3.plot(h[2])
ax3.plot(dh[2])
ax3.set_title('model loss')
ax3.set_ylabel('loss')
ax3.set_xlabel('epoch')
ax3.legend(['Conventional', 'Dopaminergic'], loc='upper right')

ax4 = pyplot.subplot(3, 2, 4)
ax4.plot(h[3])
ax4.plot(dh[3])
ax4.set_title('validation loss')
ax4.set_ylabel('loss')
ax4.set_xlabel('epoch')
ax4.legend(['Convetional', 'Dopaminergic'], loc='upper right')
'''

ax5 = pyplot.subplot(1,1,1)
ax5.plot(act, color='green')
ax5.set_title('Number of Dopaminergic Activations')
ax5.set_ylabel('Activations')
ax5.set_xlabel('epoch')
ax5.legend(['Activations'], loc='upper right')
'''
pyplot.show()







