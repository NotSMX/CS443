'''conv_layers.py
Convolutional layers in a predictive coding network (PCN)
DANIEL YU & JORDAN WANG
CS 443: Bio-Inspired Learning
'''
import tensorflow as tf

import layers


class Conv2D(layers.Layer):
    '''A 2D convolutional layer'''
    def __init__(self, name, units, kernel_size=(1, 1), strides=1, activation='relu', wt_scale=1e-3,
                 prev_layer_or_block=None, wt_init='normal', do_group_norm=False):
        '''Conv2D layer constructor.

        Parameters:
        -----------
        name: str.
            Human-readable name for the current layer (e.g. Conv2D_0). Used for debugging and printing summary of net.
        units: ints.
            Number of convolutional filters/units (K).
        kernel_size: int or tuple. len(kernel_size)=2.
            The horizontal and vertical extent (pixels) of the convolutional filters.
            These will always be the same. For example: (2, 2), (3, 3), etc.
            If user passes in an int, we convert to a tuple. Example: 3 → (3, 3)
        strides: int.
            The horizontal AND vertical stride of the convolution operation. These will always be the same.
            By convention, we use a single int to specify both of them.
        activation: str.
            Name of the activation function to apply in the layer.
        wt_scale: float.
            The standard deviation of the layer weights/bias when initialized according to a standard normal
            distribution ('normal' method).
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
        wt_init: str.
            The method used to initialize the weights/biases. Options: 'normal', 'he'.
            NOTE: Ignore until instructed otherwise.
        do_group_norm. bool:
            Whether to do group normalization in this layer.
            NOTE: Ignore until instructed otherwise.

        TODO: Set the parameters as instance variables. Call the superclass constructor to handle setting instance vars
        the child has in common with the parent class.
        '''

        # Keep me. You can specify kernel size as an int, but this snippet converts to tuple be explicitly account for
        # kernel width and height.
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        super().__init__(layer_name=name, activation=activation, prev_layer_or_block=prev_layer_or_block,
                         do_group_norm=do_group_norm)
 
        self.units = units
        self.kernel_size = kernel_size
        self.strides = strides
        self.wt_scale = wt_scale
        self.wt_init = wt_init
 

    def has_wts(self):
        '''Returns whether the Conv2D layer has weights. This is always true so always return... :)'''
        return True
    
    def get_kaiming_gain(self):
        return 1.0

    def init_params(self, input_shape):
        '''Initializes the Conv2D layer's weights and biases.

        Parameters:
        -----------
        input_shape: Python list. len(input_shape)=4.
            The anticipated shape of mini-batches of input that the layer will process: (B, Iy, Ix, K1).
            K1 is the number of units/filters in the previous layer.

        NOTE: Remember to set your wts/biases as tf.Variables so that we can update the values in the network graph
        during training.
        '''
        N, I_y, I_x, n_chans = input_shape
        kH, kW = self.kernel_size
        K = self.units
 
        if self.wt_init == 'normal':
            w_init = tf.random.normal(shape=(kH, kW, n_chans, K), stddev=self.wt_scale, dtype=tf.float32)
        elif self.wt_init == 'he':
            fan_in = kH * kW * int(n_chans)
            gain = self.get_kaiming_gain()
            std = gain * tf.sqrt(2.0 / tf.cast(fan_in, tf.float32))
            w_init = tf.random.normal(shape=(kH, kW, n_chans, K), stddev=std, dtype=tf.float32)
        else:
            raise ValueError(f'Unknown wt_init: {self.wt_init}')
 
        self.wts = tf.Variable(w_init, trainable=True, name='wts')
        self.b = tf.Variable(tf.zeros(shape=(K,), dtype=tf.float32), trainable=True, name='bias')
 
    def compute_net_input(self, x):
        '''Computes the net input for the current Conv2D layer. Uses SAME boundary conditions.

        Parameters:
        -----------
        x: tf.float32 tensor. shape=(B, Iy, Ix, K1).
            Input from the layer beneath in the network. K1 is the number of units in the previous layer.

        Returns:
        --------
        tf.float32 tensor. shape=(B, Iy, Ix, K2).
            The net_in. K2 is the number of units in the current layer.

        TODO:
        1. This layer uses lazy initialization. This means that if the wts are currently None when we enter this method,
        we should call `init_params` to initialize the parameters!
        2. Compute the convolution using TensorFlow's conv2d function. You can leave the dilations and data_format
        keyword arguments to their default values / you do not need to specify these parameters.

        Helpful link: https://www.tensorflow.org/api_docs/python/tf/nn/conv2d

        NOTE: Don't forget the bias!
        '''
        if self.wts is None:
            self.init_params(input_shape=x.shape)
        return tf.nn.conv2d(x, self.wts, strides=[1, self.strides, self.strides, 1], padding='SAME') + self.b


    def compute_group_norm(self, net_in, eps=0.001):
        '''Computes group normalization for the input tensor. Group normalization normalizes the activations among
        groups of neurons in a layer for each data point independently.

        Uses `self.num_groups` groups of neurons to perform the normalization. If the user has not set a value for
        `self.num_groups` for the current layer, we set it to the number of units in the layer divided by 8 (rounded).
        In this case when the user did not specify `self.num_groups`, we do not allow the number of groups to drop below
        8.

        (Ignore until later in the semester)

        Parameters:
        -----------
        x: tf.float32 tensor. shape=(B, Iy, Ix, K).
            Input tensor to be normalized.
        eps: float.
            A small constant added to the standard deviation to prevent division by zero. Default is 0.001.

        Returns:
        --------
        tf.float32 tensor. shape=(B, Iy, Ix, K).
            The normalized tensor with the same shape as the input tensor.
        '''
        _, Iy, Ix, K = net_in.shape

        if self.num_groups is None:
            groups = max(1, int(round(K / 8)))
        else:
            groups = int(self.num_groups)

        groups = min(groups, int(K))
        while groups > 1 and (K % groups) != 0:
            groups -= 1

        group_size = K // groups
        x_group = tf.reshape(net_in, [-1, Iy, Ix, groups, group_size])
        mean = tf.reduce_mean(x_group, axis=[1, 2, 4], keepdims=True)
        var = tf.reduce_mean(tf.square(x_group - mean), axis=[1, 2, 4], keepdims=True)
        x_norm = (x_group - mean) / tf.sqrt(var + eps)
        x_norm = tf.reshape(x_norm, [-1, Iy, Ix, K])

        if self.gn_gain is not None and self.gn_bias is not None:
            x_norm = x_norm * tf.reshape(self.gn_gain, [1, 1, 1, K]) + tf.reshape(self.gn_bias, [1, 1, 1, K])

        return x_norm


    def __str__(self):
        '''This layer's "ToString" method. Feel free to customize if you want to make the layer description fancy,
        but this method is provided to you. You should not need to modify it.
        '''
        return f'Conv2D layer output({self.layer_name}) shape: {self.output_shape}'


class MaxPool2D(layers.Layer):
    '''A 2D maxpooling layer.'''
    def __init__(self, name, pool_size=(2, 2), strides=1, prev_layer_or_block=None):
        '''MaxPool2D layer constructor.

        Parameters:
        -----------
        name: str.
            Human-readable name for the current layer (e.g. Drop_0). Used for debugging and printing summary of net.
        pool_size. tuple. len(pool_size)=2.
            The horizontal and vertical size of the pooling window.
            These will always be the same. For example: (2, 2), (3, 3), etc.
        strides. int.
            The horizontal AND vertical stride of the max pooling operation. These will always be the same.
            By convention, we use a single int to specify both of them.
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
            Example (standard MLP): Input → Dense_Hidden → Dense_Output.
                The Dense_Output Layer object has `prev_layer_or_block=Dense_Hidden`.

        TODO: Set the parameters as instance variables. Call the superclass constructor to handle setting instance vars
        the child has in common with the parent class.
        '''

        # Keep me. You can specify pool window size as an int, but this snippet converts to tuple be explicitly account
        # for window width and height.
        if isinstance(pool_size, int):
            pool_size = (pool_size, pool_size)

        super().__init__(layer_name=name, activation='linear', prev_layer_or_block=prev_layer_or_block)
        self.pool_size = pool_size
        self.strides = strides

    def compute_net_input(self, x):
        '''Computes the net input for the current MaxPool2D layer.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, Iy, Ix, K1).
            Input from the layer beneath in the network. Should be 4D (e.g. from a Conv2D or MaxPool2D layer).
            K1 refers to the number of units/filters in the PREVIOUS layer.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, Iy, Ix, K2).
            The net_in. K2 refers to the number of units/filters in the CURRENT layer.

        TODO: Compute the max pooling using TensorFlow's max_pool2d function. You can leave the data_format
        keyword arguments to its default value.

        Helpful link: https://www.tensorflow.org/api_docs/python/tf/nn/max_pool2d
        '''
        return tf.nn.max_pool2d(x, ksize=self.pool_size, strides=[1, self.strides, self.strides, 1], padding='VALID')
    
    def __str__(self):
        '''This layer's "ToString" method. Feel free to customize if you want to make the layer description fancy,
        but this method is provided to you. You should not need to modify it.
        '''
        return f'MaxPool2D layer output({self.layer_name}) shape: {self.output_shape}'


class Conv2DTranspose(Conv2D):
    '''2D transposed convolution layer'''
    def __init__(self, name, kernel_size=(1, 1), strides=1, activation='linear', wt_scale=1e-3,
                 prev_layer_or_block=None, wt_init='normal', do_group_norm=False):
        '''Conv2DTranspose layer constructor.

        Parameters:
        -----------
        name: str.
            Human-readable name for the current layer (e.g. Conv2DTranspose_0). Used for debugging and printing.
        kernel_size: int or tuple. len(kernel_size)=2.
            The horizontal and vertical extent (pixels) of the convolutional filters.
            These will always be the same. For example: (2, 2), (3, 3), etc.
            If user passes in an int, we convert to a tuple. Example: 3 → (3, 3)
        strides. int.
            The horizontal AND vertical stride of the convolution operation. These will always be the same.
            By convention, we use a single int to specify both of them.
        activation: str.
            Name of the activation function to apply in the layer.
        wt_scale: float.
            The standard deviation of the layer weights/bias when initialized according to a standard normal
            distribution ('normal' method).
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
        wt_init: str.
            The method used to initialize the weights/biases. Options: 'normal', 'he'.
            NOTE: Ignore until instructed otherwise.
        do_group_norm. bool:
            Whether to do group normalization in this layer.
            NOTE: Ignore until instructed otherwise.

        TODO: Call the superclass constructor to handle setting instance vars the child has in common with the parent
        class.

        NOTE: The units will be set during lazy initialization so set units to None for the time being.
        '''
        super().__init__(name=name, units=None, kernel_size=kernel_size, strides=strides,
                         activation=activation, wt_scale=wt_scale, prev_layer_or_block=prev_layer_or_block,
                         wt_init=wt_init, do_group_norm=do_group_norm)

    def init_params(self, input_shape):
        '''Initializes the Conv2D layer's weights and biases.

        Parameters:
        -----------
        input_shape: Python list. len(input_shape)=4.
            The anticipated shape of mini-batches of input that the layer will process: (B, Iy, Ix, K1).
            K1 is the number of units/filters in the previous layer.

        NOTE: Remember to set your wts/biases as tf.Variables so that we can update the values in the network graph
        during training.
        '''
        N, I_y, I_x, K_current = input_shape
        kH, kW = self.kernel_size
        units_prev = self.units  # already set before calling init_params
 
        if self.wt_init == 'normal':
            w_init = tf.random.normal(shape=(kH, kW, units_prev, K_current), stddev=self.wt_scale, dtype=tf.float32)
        elif self.wt_init == 'he':
            fan_in = kH * kW * K_current
            std = tf.sqrt(1.0 / tf.cast(fan_in, tf.float32))
            w_init = tf.random.normal(shape=(kH, kW, units_prev, K_current), stddev=std, dtype=tf.float32)
        else:
            raise ValueError(f'Unknown wt_init: {self.wt_init}')
 
        self.wts = tf.Variable(w_init, trainable=True, name='wts')
        self.b = tf.Variable(tf.zeros(shape=(units_prev,), dtype=tf.float32), trainable=True, name='bias')

    def compute_net_input(self, x, units_prev):
        '''Computes the net input for the current Conv2D layer. Uses SAME boundary conditions.

        Parameters:
        -----------
        x: tf.float32 tensor. shape=(B, Iy, Ix, K).
            Input from the *current* layer. K is the number of units in the *current* layer.
        units_prev: int.
            Number of units in the *previous* layer beneath the current one.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, Iy, Ix, units_prev).
            The net_in. units_prev is the number of units in the *previous* layer beneath the current one.

        TODO:
        0. Because we do not know the units in the layer below when creating the Conv2DTranspose layer, self.units
        should be None during the first forward pass. If this is the case, make sure to set the units to `units_prev`
        (remember this layer is for sending feedback).
        1. This layer uses lazy initialization. This means that if the wts are currently None when we enter this method,
        we should call `init_params` to initialize the parameters!
        2. Compute the convolution using TensorFlow's conv2d_transpose function.
        You can leave the dilations and data_format keyword arguments to their default values / you do not need to
        specify these parameters.
        3. When computing the netIn output shape, remember to account for the stride. When stride >1, this has an
        UPSCALING effect on the image. For example for a 32x32 image that is processed with conv2d_transpose and
        stride 2, the final image size will be 64x64.

        Helpful link: https://www.tensorflow.org/api_docs/python/tf/nn/conv2d_transpose

        NOTE: Don't forget the bias!
        '''
        if self.units is None:
            self.units = units_prev
        if self.wts is None:
            self.init_params(input_shape=x.shape)
 
        B_dyn = tf.shape(x)[0]
        _, I_y, I_x, K = x.shape
        out_H = int(I_y) * self.strides
        out_W = int(I_x) * self.strides
        output_shape = tf.stack([B_dyn, out_H, out_W, units_prev])
 
        net_in = tf.nn.conv2d_transpose(
            x, self.wts, output_shape=output_shape,
            strides=[1, self.strides, self.strides, 1], padding='SAME'
        ) + self.b
        return net_in


    def __call__(self, x, units_prev):
        '''Do a forward pass thru the layer with mini-batch `x`.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, Iy, Ix, K)
            The input mini-batch computed in the current layer.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, Iy, Ix, K_prev)
            The activation computed on the current mini-batch.

        TODO: This should be the same as the one in Layer, except note the difference in method signature for
        compute_net_input.
        '''
        net_in = self.compute_net_input(x, units_prev)
        if self.do_group_norm and self.gn_gain is not None:
            net_in = self.compute_group_norm(net_in)
        net_act = self.compute_net_activation(net_in)
        import math
        _, Iy, Ix, _ = x.shape
        h = math.ceil(int(Iy) / self.strides)
        w = math.ceil(int(Ix) / self.strides)
        self.output_shape = [net_act.shape[0], h, w, units_prev]
        return net_act


    def __str__(self):
        '''This layer's "ToString" method. Feel free to customize if you want to make the layer description fancy,
        but this method is provided to you. You should not need to modify it.
        '''
        return f'Conv2DTranspose layer output({self.layer_name}) shape: {self.output_shape}'
