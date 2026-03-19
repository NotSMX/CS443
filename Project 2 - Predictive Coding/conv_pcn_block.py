'''conv_pcn_block.py
The convolutional predictive coding block for the ConvPCN.
YOUR NAMES HERE
CS 443: Bio-Inspired Learning
'''
import tensorflow as tf

import block
from layers import Dropout
from conv_layers import Conv2D, Conv2DTranspose


class ConvPCNBlock(block.Block):
    '''The convolutional predictive coding block, the main building block in the ConvPCN.

    It is composed of a:
    - `Conv2DTranspose` layer for sending feedback signals backwards. Uses ReLU activation.
    - `Conv2D` layer for sending feedforward signals forward. Uses ReLU activation.
    - (optional) `Dropout` layer
    '''
    def __init__(self, blockname, units, kernel_size, strides, num_steps, state_lr, dropout_rate, wt_init,
                 do_group_norm, prev_layer_or_block):
        '''ConvPCNBlock constructor.

        Parameters:
        -----------
        blockname: str.
            Human-readable name for the current block (e.g. ConvPCNBlock_0). Used for debugging and printing summary of
            net.
        units: ints.
            Number of convolutional filters/units in the Conv2D layer in the block (K).
        kernel_size: int or tuple. len(kernel_size)=2.
            The horizontal and vertical extent (pixels) of the convolutional filters.
            These will always be the same. For example: (2, 2), (3, 3), etc.
            If user passes in an int, we convert to a tuple. Example: 3 → (3, 3)
        strides: int.
            The horizontal AND vertical stride of the convolutional operations. These will always be the same.
            By convention, we use a single int to specify both of them.
        num_steps: int.
            Number of steps to use to update the block's state during each forward pass thru the block.
        state_lr: float.
            The learning rate used to iteratively update the block's state during each forward pass thru the block.
        dropout_rate: float or None.
            The dropout rate used in the `Dropout` layer in the block.
            If `None` passed in, then we do not use `Dropout` in the block.
        wt_init: str.
            The method used to initialize the weights/biases. Options: 'normal', 'he'.
        do_group_norm. bool:
            Whether to do group normalization in this layer.
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.

        TODO:
        1. Call the superclass constructor to pass in shared parameters.
        2. Build out and configure layers that belong to the block and add them to the self.layers list.
        3. Create instance variables for other parameters as needed.
        '''
        self.layers = []

    def __call__(self, x):
        '''Forward pass through the ConvPCNBlock.

        Parameters:
        -----------
        x: tf.float32 tensor. shape=(B, Iy, Ix, K_prev).
            Input from the layer beneath in the network. K_prev is the number of units in the previous layer.

        Returns:
        --------
        tf.float32 tensor. shape=(B, Iy, Ix, K).
            The evolving state established from iterative bottom-up and top-down interactions the block. K is the number
            of units in the block (i.e. the Conv2D layer).

        TODO:
        1. Establish the initial state of the block by computing the netActs in the 2D conv layer.
        2. Use the 2D transposed conv layer to reconstruct the block input.
        3. Compute the prediction error (see notebook for refresher on equation).
        4. Use the 2D conv layer to forward the prediction error back into the block then use that to update the block's
        state (see notebook for refresher on equation).
        5. Repeat steps 2-4 for the preset number of steps.
        6. Apply the dropout (if we are doing).
        '''
        pass