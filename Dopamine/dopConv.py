from tf.keras.engine.input_spec import InputSpec










class Conv(Layer):

	def __init__(self,rank,filters,kernel_size,strides=1,padding='valid',activation=None,kernel_initializer='glorot_uniform', bias_initializer='zeros',trainable=True,name=None,conv_op=None,**kwargs):
               
	super(Conv, self).__init__(trainable=trainable,name=name,**kwargs)
	self.rank = rank

	if isinstance(filters, float):
		filters = int(filters)
	if filters is not None and filters < 0:
		raise ValueError(f'Received a negative value for `filters`. Was expecting a positive value. Received {filters}.')
      
      
	self.filters = filters
	self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
	self.strides = conv_utils.normalize_tuple(strides, rank, 'strides', allow_zero=True)
	self.padding = conv_utils.normalize_padding(padding)
	self.data_format = conv_utils.normalize_data_format('channels_last')

	self.activation = activations.get(activation)

	self.kernel_initializer = initializers.get(kernel_initializer)
	self.bias_initializer = initializers.get(bias_initializer)
	self.input_spec = InputSpec(min_ndim=self.rank + 2)

	self._validate_init()
	self._tf_data_format = conv_utils.convert_data_format(self.data_format, self.rank + 2)


	def _validate_init(self):

		if not all(self.kernel_size):
			raise ValueError('The argument `kernel_size` cannot contain 0(s). Received: %s' % (self.kernel_size,))

		if not all(self.strides):
			raise ValueError('The argument `strides` cannot contains 0(s). Received: %s' % (self.strides,))

 


	def build(self, input_shape):
		input_shape = tf.TensorShape(input_shape)
		input_channel = self._get_input_channel(input_shape)
		kernel_shape = self.kernel_size + (input_channel // self.filters)

		# compute_output_shape contains some validation logic for the input shape,
		# and make sure the output shape has all positive dimentions.
		self.compute_output_shape(input_shape)

		self.kernel = self.add_weight(name='kernel',shape=kernel_shape,initializer=self.kernel_initializer,trainable=True,dtype=self.dtype)
		
		
		self.bias = self.add_weight(name='bias',shape=(self.filters,),initializer=self.bias_initializer,trainable=True,dtype=self.dtype)
			
		
		channel_axis = self._get_channel_axis()
		self.input_spec = InputSpec(min_ndim=self.rank + 2,axes={channel_axis: input_channel})
		
		self.built = True


	def convolution_op(self, inputs, kernel):
		if isinstance(self.padding, str):
			tf_padding = self.padding.upper()
		else:
			tf_padding = self.padding

		return tf.nn.convolution(inputs,kernel,strides=list(self.strides),padding=tf_padding,data_format=self._tf_data_format,name=self.__class__.__name__)

	def _spatial_output_shape(self, spatial_input_shape):
		return [conv_utils.conv_output_length(length,self.kernel_size[i],padding=self.padding,stride=self.strides[i]) 
			for i, length in enumerate(spatial_input_shape)]


	def compute_output_shape(self, input_shape):
		input_shape = tf.TensorShape(input_shape).as_list()
		batch_rank = len(input_shape) - self.rank - 1
		try:
				return tf.TensorShape(input_shape[:batch_rank] + self._spatial_output_shape(input_shape[batch_rank:-1]) +[self.filters])
			


	def _get_input_channel(self, input_shape):
		channel_axis = self._get_channel_axis()
		if input_shape.dims[channel_axis].value is None:
			raise ValueError('The channel dimension of the inputs should be defined. '
					f'The input_shape received is {input_shape}, '
					f'where axis {channel_axis} (0-based) '
					'is the channel dimension, which found to be `None`.')
		return int(input_shape[channel_axis])


	def call(self, inputs):
		input_shape = inputs.shape
		outputs = self.convolution_op(inputs, self.kernel)		
		output_rank = outputs.shape.rank
		

		if output_rank is not None and output_rank > 2 + self.rank:

			def _apply_fn(o):
				return tf.nn.bias_add(o, self.bias, data_format=self._tf_data_format)

			outputs = conv_utils.squeeze_batch_dims(outputs, _apply_fn, inner_rank=self.rank + 1)
		else:
			outputs = tf.nn.bias_add(outputs, self.bias, data_format=self._tf_data_format)

		#if not tf.executing_eagerly():
		# Infer the static output shape:
		#	out_shape = self.compute_output_shape(input_shape)
		#	outputs.set_shape(out_shape)

		if self.activation is not None:
			return self.activation(outputs)
		return outputs


    
	

    


















































