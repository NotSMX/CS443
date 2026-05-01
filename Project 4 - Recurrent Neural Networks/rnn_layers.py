'''rnn_layers.py
New layers specific to the RNN network
YOUR NAMES HERE
CS 443: Bio-inspired Machine Learning
Project 4: Recurrent Neural Networks
'''
import tensorflow as tf

import layers


class GRU(layers.Layer):
    '''Gated Recurrent Unit (GRU) layer.
    '''
    def __init__(self, name, units, prev_layer_or_block=None):
        '''GRU constructor.

        Parameters:
        -----------
        name: str.
            Human readable name for the layer.
        units. int.
            Number of GRUs/neurons H to create in the layer.
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.

        TODO: Call the superclass constructor, filling in all relevant information. Assign any additional parameters as
        instance vars as needed.
        '''

        # Wts and bias placeholders
        # Update gate related wts/bias
        self.wts_update_i2h = None
        self.wts_update_h2h = None
        self.update_b = None

        # Reset gate related wts/bias
        self.wts_reset_i2h = None
        self.wts_reset_h2h = None
        self.reset_b = None

        # Candidate gate related wts/bias
        self.wts_cand_i2h = None
        self.wts_cand_h2h = None
        self.cand_b = None

    def has_wts(self):
        '''Returns whether the layer has weights. This is always true so always return... :)'''
        return True

    def save_wts(self):
        '''Create a dictionary with all the weights and biases from the layer in NumPy ndarray format for saving to
        disk or analysis.

        (This method is provided to you. It should not requre modification.)

        Returns:
        --------
        Python dictionary.
            The weights and biases from the layer in NumPy ndarray format.
        '''
        params = {}
        params['wts_update_i2h'] = self.wts_update_i2h.numpy()
        params['wts_update_h2h'] = self.wts_update_h2h.numpy()
        params['update_b'] = self.update_b.numpy()
        params['wts_reset_i2h'] = self.wts_reset_i2h.numpy()
        params['wts_reset_h2h'] = self.wts_reset_h2h.numpy()
        params['reset_b'] = self.reset_b.numpy()
        params['wts_cand_i2h'] = self.wts_cand_i2h.numpy()
        params['wts_cand_h2h'] = self.wts_cand_h2h.numpy()
        params['cand_b'] = self.cand_b.numpy()

        return params

    def load_wts(self, params):
        '''Replaces weights and biases in the layer by those in `params`.

        (This method is provided to you. It should not requre modification.)

        Parameters:
        -----------
        params: Python dictionary.
            The weights and biases from the layer in NumPy ndarray format.
        '''
        self.wts_update_i2h.assign(params['wts_update_i2h'])
        self.wts_update_h2h.assign(params['wts_update_h2h'])
        self.update_b.assign(params['update_b'])
        self.wts_reset_i2h.assign(params['wts_reset_i2h'])
        self.wts_reset_h2h.assign(params['wts_reset_h2h'])
        self.reset_b.assign(params['reset_b'])
        self.wts_cand_i2h.assign(params['wts_cand_i2h'])
        self.wts_cand_h2h.assign(params['wts_cand_h2h'])
        self.cand_b.assign(params['cand_b'])

    def init_params(self, input_shape):
        '''Initializes the GRU layer's weights and biases.

        Parameters:
        -----------
        input_shape: Python list.
            The anticipated shape of mini-batches of input that the layer will process.
            Format: (B, T, H_prev), where B is the mini-batch size, T is the sequence length, and H_prev is the number
            of units in the previous layer below.

        NOTE:
        1. See the constructor for naming scheme for the weights/biases.
        2. Remember to set your wts/biases as tf.Variables so that we can update the values in the network graph during
        training.
        3. For consistency with the test code, initialize your wts in this order: update gates, reset gates, candidate.
        4. Use He/Kaiming initialization. See the notes and notebook for refreshers on the gains and the strategy for
        the hidden-to-hidden weights.
        '''
        pass

    def get_wts(self):
        '''Return all the weights in the layer in a Python list.

        (This method is provided to you. It should not requre modification.)
        '''
        all_wts = [self.wts_update_i2h, self.wts_update_h2h, self.wts_reset_i2h, self.wts_reset_h2h, self.wts_cand_i2h,
                self.wts_cand_h2h]
        return all_wts

    def get_b(self):
        '''Return all the biases in the layer in a Python list.

        (This method is provided to you. It should not requre modification.)
        '''
        all_b = [self.update_b, self.reset_b, self.cand_b]
        return all_b

    def get_params(self):
        '''Return both the weights and biases in the layer in a Python list.

        (This method is provided to you. It should not requre modification.)
        '''
        params = []
        params.extend(self.get_wts())
        params.extend(self.get_b())
        return params

    def compute_net_input(self, x, state):
        '''Computes the net input of the GRU Layer for the current time step.

        Parameters:
        -----------
        x: tf.float32 tensor. shape=(B, H_prev).
            Input signal produced by the layer below for the current time step.
        state: tf.float32 tensor. shape=(B, H).
            The current GRU state/evolving net_act from the previous time step.

        Returns:
        -----------
        tf.float32 tensor. shape=(B, H).
            Net input for the update gate of all units in the GRU layer.
        tf.float32 tensor. shape=(B, H).
            Net input for the reset gate of all units in the GRU layer.
        tf.float32 tensor. shape=(B, H).
            Net input for the candidate act of all units in the GRU layer.

        NOTE:
        - Don't forget that there are both feedforward AND recurrent connections in this layer.
        - Don't forget to defer a component of the GRU y candidate netIn
        '''
        if self.wts_update_i2h is None:
            self.init_params(input_shape=x.shape)

        pass

    def compute_net_activation(self, update_gate_in, reset_gate_in, cand_in, state):
        '''Computes the state and net activation of the GRU Layer for the current time step.

        Parameters:
        -----------
        update_gate_in: tf.float32 tensor. shape=(B, H).
            Net input for the update gate of all units in the GRU layer.
        reset_gate_in: tf.float32 tensor. shape=(B, H).
            Net input for the reset gate of all units in the GRU layer.
        cand_in: tf.float32 tensor. shape=(B, H).
            Net input for the candidate act of all units in the GRU layer.
        state: tf.float32 tensor. shape=(B, H).
            The current GRU state/evolving net_act from the previous time step.

        Returns:
        --------
        tf.float32 tensor. shape=(B, H).
            The GRU state/evolving net_act computed for the current time step.
        tf.float32 tensor. shape=(B, H).
            The update gate net_act computed for the current time step.
        tf.float32 tensor. shape=(B, H).
            The reset gate computed for the current time step.
        '''
        pass

    def reset_state(self, B):
        '''Returns the reset/default GRU state of 0s for all neurons.

        Parameters:
        -----------
        B: int.
            Number of samples in the mini-batch.

        Returns:
        --------
        tf.float32 tensor. shape=(B, H).
            The reset/default GRU state of 0s for all neurons.
        '''
        pass

    def __call__(self, x, mask, state=None):
        '''Do a forward pass thru the GRU layer with mini-batch `x`.

        Parameters:
        -----------
        x: tf.float32 tensor. shape=(B, T, H_prev).
            The input signal from the layer below that contains `H_prev` units.
        mask: tf.float32 tensor. shape=(B, T, 1).
            Padding mask for time step of each sample in the mini-batch. Values are binary:
            1 if the current token is NOT the padding token.
            0 if the current token IS the padding token.
            We do NOT allow the evolving state to be corrupted/updated at any padding token. Instead, in these cases,
            we use the mask to keep the PREVIOUS GRU state.
        state: tf.float32 tensor. shape=(B, H).
            Initial state to use when processing the mini-batch. If nothing is passed in (`None`), we start with a
            reset state.

        Returns:
        --------
        tf.float32 tensor. shape=(B, T, H).
            The history of states/netAct values at each time step of the mini-batch.

        NOTE:
        1. "Unroll" the input and process the input by the layer sequentially across time.
        2. tf.stack could be helpful to aggregating the states recorded at each time step...
        https://www.tensorflow.org/api_docs/python/tf/stack
        3. Before the method ends, check to see if `self.output_shape` is None. If it is, that means we are processing
        our very first mini-batch of data ever (e.g. at the beginning of training). If `self.output_shape` is None,
        set it to the shape of the layer's 3D activations, represented as a Python list. You can convert something into
        a Python list by calling the `list` function — e.g. `list(blah)`.
        '''
        B, T, H_prev = x.shape


    def __str__(self):
        '''This layer's "ToString" method. Feel free to customize if you want to make the layer description fancy,
        but this method is provided to you. You should not need to modify it.
        '''
        return f'GRU layer output({self.layer_name}) shape: {self.output_shape}'
