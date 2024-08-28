import keras
import numpy as np

from keras.layers import Input, Dense, Multiply
from keras.models import Model

def generate_mul_samples(n):

	#Real Data
	X1 = np.random.rand(n) - 0.5
	X2 = X1 * X1 * X1
	X1 = X1.reshape(n,1)
	X2 = X2.reshape(n,1)
	d1 = np.hstack((X1, X2))
	y1 = np.zeros((n, 1))
	
	X1 = np.random.rand(n) - 0.5
	X2 = X1 * X1
	X1 = X1.reshape(n,1)
	X2 = X2.reshape(n,1)
	d2 = np.hstack((X1, X2))
	y2 = np.ones((n, 1))
	
		
	return (X1, X2)
	
	
def generate_sum_samples(n):

	#Real Data
	X1 = np.random.rand(n) - 0.5
	X2 = X1 + X1 + X1
	X1 = X1.reshape(n,1)
	X2 = X2.reshape(n,1)
	d1 = np.hstack((X1, X2))
	y1 = np.zeros((n, 1))
	
	X1 = np.random.rand(n) - 0.5
	X2 = X1 + X1
	X1 = X1.reshape(n,1)
	X2 = X2.reshape(n,1)
	d2 = np.hstack((X1, X2))
	y2 = np.ones((n, 1))
	
		
	return (X1, X2)


task = Input(shape=(3,))
m1 = Dense(25, activation='relu', kernel_initializer='he_uniform')(data)
q_out = Dense(1, activation='sigmoid')(m1)
mask_model = Model(inputs=task, outputs=q_out)

data = Input(shape=(2,))
h1 = Dense(25, activation='relu', kernel_initializer='he_uniform')(data)
mul_1 = Multiply()([h1, m1])
out = Dense(1, activation='sigmoid')(mul_1)
main_model = Model(inputs=data, outputs=out)

main_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
mask_model.compile(loss='mse', optimizer='adam', metrics=['mae'])




