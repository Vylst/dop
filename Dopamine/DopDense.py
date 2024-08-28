import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Layer
from matplotlib import pyplot

tf.config.run_functions_eagerly(True)

epoch_variable = np.array(0, dtype=np.int32)
cnt = np.zeros((300))

class MyCallback(Callback):
	def __init__(self):
		self.epoch_variable = epoch_variable
		
	def on_epoch_begin(self, epoch, logs=None):
		self.epoch_variable += 1
		
	def on_training_end(self, logs=None):
		print("Dopaminergic activations: \n")
		print('[', end='')
		for i in range(len(cnt)):
			print(cnt[i], ',', end='')
		print(']')
		
		ax = pyplot.subplot(1,1,1)
		ax.plot(cnt[0:150], color='green')
		ax.set_title('Number of Dopaminergic Activations')
		ax.set_ylabel('Activations')
		ax.set_xlabel('epoch')
		ax.legend(['Activations'], loc='upper right')

class DopDense(Layer):

	def __init__(self, activation, units=32, n_dop=None, threshold=0, refractory_period=0):
		'''Initializes the instance attributes'''
		super(DopDense, self).__init__()
		self.units = units			#numero total de neuronios na camada
		self.threshold = threshold		#efeito dopaminergico so aplicado se este threshold for ultrapassado 
		self.ref_period = refractory_period	#Periodo de intervalo que um neuronio tem de esperar ate poder voltar a aplicar efeito dopaminergico sobre ligaçoes
							#Se ndop foi ativado na iteraçao x e há um periodo refratorio de y, então só pode voltar a ativar-se em x+y
		self.activation = activation		#função de ativação do neuronio
		
		#O numero de neuronios dopaminergicos tem de ser menor ou igual ao numero total de neuronios na camada
		if(n_dop == None):
			self.n_dop = np.random.randint(1, units/2)	
		else:
			if(n_dop <= units/2):			
				self.n_dop = n_dop	
			else:
				raise ValueError("Dopaminergic neurons must not exceed half of layer size.")
				
		## Generate indices for dopaminergic neurons
		self.dop_indices = np.linspace(1, self.units-1, self.n_dop, dtype=np.int32)

		## Placeholders for dopaminergic neuron weights
		self.dop_weights_new = tf.Variable(0., trainable=False)
		self.dop_weights_old = tf.Variable(0., trainable=False)

		## Initialize refraction period indicator for each dopaminergic neuron  --->  Cada neuronio dopaminergico tem um counter para ver se esta ou nao num periodo refratorio
		self.indicator = -self.ref_period*np.ones((self.n_dop))


	def build(self, input_shape):
		'''Create the state of the layer (weights)'''        

		self.batch_ctr = 0


		## Number of connections per neuron in this layer, with neurons from the previous layer
		self.n_inputs = input_shape[-1]

		## initialize the weights
		w_init = tf.random_normal_initializer()
		self.w = tf.Variable(name="kernel", initial_value=w_init(shape=(input_shape[-1], self.units), dtype='float32'), trainable=True)

		## initialize the biases
		b_init = tf.zeros_initializer()
		self.b = tf.Variable(name="bias", initial_value=b_init(shape=(self.units,), dtype='float32'), trainable=True)
		
		## Dopaminergic neuron filter		---> Mascara com 1s nos indices correspondentes aos neuronios dopaminergicos
		f = np.zeros((self.n_dop, input_shape[-1], self.units), dtype=np.float32)
		for i in range(self.n_dop):
			f[i,:,self.dop_indices[i]] = 1
		self.filter = tf.Variable(f, dtype=tf.float32, trainable=False)
		

	## Check if training or testing
	def train_test(self):
		return tf.keras.backend.in_train_phase(1, 0)
	
	def call(self, inputs):
		'''Defines the computation from inputs to outputs'''

		#If training
		if(self.train_test()):
			## Gather weights from dopaminergic neurons
			self.dop_weights_new = tf.math.multiply(self.w, tf.reduce_sum(self.filter,axis=0))
			
			
			#tf.print("\n\n\nepoch: ", epoch_variable, " batch: ", self.batch_ctr)
			#tf.print("Dopaminergic neurons: ", self.dop_indices, "\n")
			#tf.print("Number of dop neurons: ", self.n_dop, "\n")
			#tf.print("\nweights:\n", self.w, summarize=-1)

			weight_multiplier = tf.Variable(np.ones((np.shape(inputs)[-1], self.units), dtype=np.float32), trainable=False)
			if(self.batch_ctr > 1):
				
				## Compute differences in dopaminergic weights between batch t and t-1, and average by number of dopaminergic connections per dopaminergic neuron
				dop_neuron_mean_difference = tf.math.reduce_sum( tf.math.abs( tf.math.subtract(self.dop_weights_new, self.dop_weights_old) ), axis=0) / self.n_inputs
				

				#tf.print("\nweights_new\n", self.dop_weights_new, summarize=-1)
				#tf.print("\nweights_old\n", self.dop_weights_old, summarize=-1)
				#tf.print("\ndifferences\n", tf.math.subtract(self.dop_weights_new, self.dop_weights_old), summarize=-1)
				#tf.print("\nmean_differences\n", dop_neuron_mean_difference, summarize=-1)

				c = 0
				for i in range(self.n_dop):
				
					## For each dopaminergic neuron, confirm if its temporal delta is greater than threshold and also if the neuron is within a refractory period 
					if( dop_neuron_mean_difference[self.dop_indices[i]] > self.threshold  and self.batch_ctr-self.indicator[i] > self.ref_period):
						'''
						## Make sure the rolled filter does not encompass neighboring dopaminergic neurons (e.g. dop neuron 10 can't impact dop neuron 11 and vice-versa)
						aux_factor = 0;
						if(self.dop_indices[i]-1 not in self.dop_indices):
							aux_factor += tf.roll(self.filter[i],-1,1)
						if(self.dop_indices[i]+1 not in self.dop_indices):
							aux_factor += tf.roll(self.filter[i],1,1)
						
						weight_multiplier = weight_multiplier * ( 1 + (10*dop_neuron_mean_difference[self.dop_indices[i]]) * aux_factor )
						'''
						#tf.print("\nDopNeuron ", self.dop_indices[i], " is activating with delta ", dop_neuron_mean_difference[self.dop_indices[i]], ".\n")
						
						
						#tf.print("\nAux_factor: \n", aux_factor, summarize=-1 )
						#tf.print("\nwm:\n", weight_multiplier, summarize=-1)
						#tf.print("\nindicator: ", self.indicator[i], "  epoch_variable: ", epoch_variable)
						
						
						## Register that this dopaminergic neuron just activated on this batch
						self.indicator[i] = self.batch_ctr
						c = c + 1

				
				#self.cntr[epoch_variable-1] = self.cntr[epoch_variable-1] + c
				cnt[epoch_variable-1] = cnt[epoch_variable-1] + c
			
			## Store current weights as old weights for next batch
			self.dop_weights_old = self.dop_weights_new
			
			## Increment batch counter
			self.batch_ctr = self.batch_ctr + 1
			
			#(self.w).assign(self.w*weight_multiplier)

		
		## Usual linear combination of fully connected layers
		#if(self.activation == "relu"):
		#	return tf.keras.activations.relu(tf.matmul(inputs, self.w) + self.b)
		#elif(self.activation == "softmax"):
		#	return tf.keras.activations.softmax(tf.matmul(inputs, self.w) + self.b)
		#else:
		return tf.matmul(inputs, self.w) + self.b
	
	
	

