'''rnn.py
Recurrent neural networks (RNNs) for text generation
YOUR NAMES HERE
CS 443: Bio-inspired Machine Learning
Project 4: Recurrent Neural Networks
'''
import numpy as np
import tensorflow as tf

import network
from layers import Dense, Dropout
from skipgram_layers import Embedding
from rnn_layers import GRU

from tf_util import arange_index

class RNN(network.DeepNetwork):
    '''Parent class for all specific types of recurrent neural networks (RNNs).
    '''
    def __init__(self, input_feats_shape, C, pad_token=0, start_token=1, end_token=2):
        '''RNN constructor.

        Parameters:
        -----------
        input_feats_shape: tuple.
            The shape of input data WITHOUT the batch dimension.
            For the RNN this is (T, M), where T is the seq len and M is the vocab size.
        C: int.
            Number of classes in the dataset / vocab size.
        pad_token: int.
            Int code for the padding token.
        start_token: int.
            Int code for the start token.
        end_token: int.
            Int code for the end token.

        TODO:
        1. Call the superclass constructor to pass along parameters that `DeepNetwork` has in common.
        2. Create instance variables for parameters as needed.
        '''

        # KEEP THE FOLLOWING
        # This is a list of boolean values that has length = layers in the net.
        # It holds True if the i-th layer in the network if a recurrent layer, False if not.
        # This is intended to be used to determine during the net's forward pass whether recurrent layers should
        # bootstrap off of an existing state, if it is provided.
        self.is_recurrent_layer = []

    def loss(self, out_net_act, y, mask, eps=1e-8):
        '''Computes the temporal cross entropy loss for the current minibatch based on the output layer activations
        `out_net_act` and int-coded class labels `y`, and padding mask `mask`.

        Parameters:
        -----------
        output_layer_net_act: tf.float32 tensor. shape=(B, T, C).
            Activation in the output layer for all time steps and sequence in the current mini-batch.
        y: tf.int32 tensor. shape=(B, T).
            int-coded next tokens in the vocabulary for every time step in each sequence in the current mini-batch.
        mask: tf.float32 tensor. shape=(B, T, 1).
            Padding mask for time step of each sample in the mini-batch. Values are binary:
            1 if the current token is NOT the padding token.
            0 if the current token IS the padding token.
            We do NOT allow activations at padding chars to count toward the overall loss (since predicting that the
            next char is a padding char is silly!!).
        eps: float.
            Small value to prevent possibly taking the log of 0 or dividing by 0.

        Returns:
        -----------
        float.
            The loss.

        NOTE:
        1. You only need to support the loss type `'temporal_cross_entropy'`. Throw an error if the user is not using
        this loss for the RNN.
        2. Use your regular cross entropy loss code as a starting point for the temporal version.
        3. Note that `arange_index` assumes netAct is 2D and y labels are 1D. Make shape adjustments to make this
        happen...
        4. Use the mask to "gate" or only count loss contributions from sequence time steps where the current token is
        NOT the padding token. Be careful when averaging the loss to normalize by the total non-padding contributions
        ONLY.
        5. If you are using tf.reshape, note that if you going to use @tf.function when training your net
        (likely you are), it likes the desired shape to be wrapped in brackets. For example, tf.reshape(blah, [-1])
        rather than tf.reshape(blah, -1).

        '''
        if self.loss_name == 'temporal_cross_entropy':
            pass
        else:
            raise ValueError(f'Unknown loss function {self.loss_name}')

        return loss

    def __call__(self, x, mask=None, states=None):
        '''Forward pass through the RNN with the data samples `x`.

        Parameters:
        -----------
        x: tf.float32 tensor. shape=(B, T).
            Mini-batch of input sequences.
        mask: tf.float32 tensor. shape=(B, T, 1) or None.
            Padding mask for time step of each sample in the mini-batch. Values are binary:
            1 if the current token is NOT the padding token.
            0 if the current token IS the padding token.
            We do NOT allow activations at padding chars to count toward the overall loss (since predicting that the
            next char is a padding char is silly!!).
        states: tuple of tf.float32 tensors or None. len=num_recurrent_layers. shape of each entry/tensor: (B, H).
            The latest state in each recurrent layer (e.g. GRU) of the net.
            If None, every recurrent layer should process starting from reset states.

        Returns:
        --------
        tf.float32 tensor. shape=(B, T, C).
            Activation in the output layer for all time steps and sequence in the current mini-batch.
        tuple of tf.float32 tensors. len=num_recurrent_layers. shape of each entry/tensor: (B, H).
            List of final states of each recurrent layer at the end of processing the current mini-batch, converted to
            a tuple.

        NOTE:
        1. Clearly, only recurrent layers should be supplied with the mask and prior states...
        2. Make use of the self.is_recurrent_layer list. It will tell you if the current layer is a recurrent layer or
        not.
        '''
        # Should just be hit when doing pilot forward pass. Otherwise, should always be created in train_step/test_step
        # By default, we don't mask out any states
        if mask is None:
            mask = tf.ones([1, x.shape[1], 1], dtype=tf.float32)

        rec_layer_states = []


        return net_act, tuple(rec_layer_states)

    @tf.function
    def train_step(self, x_batch, y_batch):
        '''Completely process a single mini-batch of data during training. This includes:
        1. Performing a forward pass of the data through the entire network.
        2. Computing the loss.
        3. Updating the network parameters using backprop (via update_params method).

        Parameters:
        -----------
        x_batch: tf.float32 tensor. shape=(B, ...).
            A single mini-batch of data packaged up by the fit method.
        y_batch: tf.ints32 tensor. shape=(B,).
            int-coded labels of samples in the mini-batch.

        Returns:
        --------
        float.
            The loss.
        '''
        # Make mask for padding char: 1 if NOT the padding char, 0 if it IS the padding char
        # mask shape: (B, T) -> (B, T, 1) for compatibility with (B, T, H) in rec layers
        mask = tf.expand_dims(tf.cast(x_batch != self.pad_int, dtype=tf.float32), axis=-1)

        # Do forward pass with gradients tracked in the tape
        with tf.GradientTape() as tape:
            net_act, _ = self(x=x_batch, mask=mask)
            loss = self.loss(net_act, y_batch, mask)

        # Do wt update
        self.update_params(tape, loss)
        return loss

    @tf.function
    def test_step(self, x_batch, y_batch):
        '''Completely process a single mini-batch of data during test/validation time. This includes:
        1. Performing a forward pass of the data through the entire network.
        2. Computing the loss.
        3. Obtaining the predicted classes for the mini-batch samples.
        4. Compute the accuracy of the predictions.

        Parameters:
        -----------
        x_batch: tf.float32 tensor. shape=(B, ...).
            A single mini-batch of data packaged up by the fit method.
        y_batch: tf.ints32 tensor. shape=(B,).
            int-coded labels of samples in the mini-batch.

        Returns:
        --------
        float.
            The accuracy.
        float.
            The loss.
        '''
        # Make mask for padding char: 1 if NOT the padding char, 0 if it IS the padding char
        # mask shape: (B, T) -> (B, T, 1) for compatibility with (B, T, H)
        mask = tf.expand_dims(tf.cast(x_batch != self.pad_int, dtype=tf.float32), axis=-1)

        # Validation loss
        # compute validation net_act
        out_net_act_val, _ = self(x=x_batch, mask=mask)
        # compute validation loss
        loss = self.loss(out_net_act_val, y_batch, mask)
        acc = tf.constant(0.0)
        return acc, loss

    def generate(self, prompt, length, char2ind_map, ind2char_map, r_seed=0):
        '''Generates/predicts a sequence of chars of length `length` chars that follow the provided prompt.
        It is helpful remember that the RNN generates chars one at a time sequentially. Therefore in
        prediction/generation mode, the network processes tokens in mini-batches of one item for one time step.

        Parameters:
        -----------
        prompt: str.
            Chars to pass thru the RNN one-at-a-time sequentially to build up the state before the net predicts the
            next char.
        length: int.
            Maximum number of chars that RNN generates after the prompt chars.
            NOTE: The RNN can decide to terminate the text generation early itself if it predicts the <END> token.
        char2ind_map: Python dictionary.
            Keys: chars in vocab. Values: int code of a char in the vocab.
        ind2char_map: Python dictionary.
            Keys: int code of a char in the vocab. Values: Which char it corresponds to in the vocab.
        r_seed: int.
            Random seed to control the randomness a NumPy RNG object that controls the sampling of which token to
            predict next based on the RNN's output layer softmax probabilities.

        Returns:
        --------
        str. len=(len(prompt) + length).
            The provided prompt concatenated with the set of RNN generated chars.

        TODO: Fill in code snippets in the places below marked with a TODO item.
        '''
        # RNG object to control which next token is probablistically selected based on the softmax probs
        rng = np.random.default_rng(r_seed)

        '''1: Convert prompt from str to int'''
        # We want to convert prompt (str) to B, T, M (where B = 1)
        # Convert chars -> int
        prompt_int = [char2ind_map[char] for char in prompt]  # (N_gen,)

        '''2: Warm up RNN state with the prompt'''
        # Warm up RNN to develop GRU state
        # a. Handle 1st token, which we assume will always be the start token
        _, states = self(tf.reshape(tf.constant(self.start_int, dtype=tf.int32), [1, 1]))
        # b. TODO: Process the rest of the prompt, except last prompt token. Allow states to progressively build.

        '''3: Generate new chars using a feedback loop (prev pred = next input), starting with last prompt char'''
        seq_gen_int = [prompt_int[-1]]
        for t in range(length):
            curr_token = seq_gen_int[-1]

            # Convert int into Tensor format (1, 1)
            x_int_tf = tf.reshape(tf.constant(curr_token, dtype=tf.int32), [1, 1]) # B, T

            # TODO: Compute net_act (B=1, T=1, vocab_sz) at output layer and updated recurrent layer states for the
            # current token

            # TODO: Extract the softmax probs at the final time step

            # Draw predicted char index from vocab proportional to the softmax prob
            pred_char_int = rng.choice(np.arange(len(out_probs_np)), p=out_probs_np)
            # Release int from numpy
            pred_char_int = pred_char_int.item()

            # TODO: Get out of loop if net decides it is done / time to end the generated sequence

            # TODO: Append to generated seq, the char corresponding to the predicted index in the vocab

        '''4: TODO: Convert the generated int tokens to chars'''

        '''5: TODO: Concat the prompt and the generated seq'''

        return generated_text


class GRU_RNN1Mini(RNN):
    '''Mini recurrent neural network with a single GRU layer.

    Embedding → GRU → Dense

    Both the input and output layer have `vocab_sz` units. The output layer uses regular softmax activation.

    All layers use He/Kaiming weight initialization.
    '''
    def __init__(self, input_feats_shape, C, embedding_dim=64, rnn_units=128):
        '''GRU_RNN1Mini constructor

        Parameters:
        -----------
        input_feats_shape: tuple.
            The shape of input data WITHOUT the batch dimension.
            For the RNN this is (T, M), where T is the seq len and M is the vocab size.
        C: int.
            Number of classes in the dataset / vocab size.
        embedding_dim: int.
            Number of neurons in the Embedding layer (M).
        rnn_units: int.
            Number of neurons in the GRU layer (H).

        TODO:
        1. Call the superclass constructor to pass along parameters that `DeepNetwork` has in common.
        2. Build out the network like usual. NOTE: you should populate the self.is_recurrent_layer list.
        '''
        pass


class GRU_RNN1(GRU_RNN1Mini):
    '''Recurrent neural network with a single GRU layer.

    Embedding → GRU → Dense

    Both the input and output layer have `vocab_sz` units. The output layer uses regular softmax activation.

    All layers use He/Kaiming weight initialization.
    '''
    def __init__(self, input_feats_shape, C, embedding_dim=64, rnn_units=256):
        '''GRU_RNN1 constructor

        Parameters:
        -----------
        input_feats_shape: tuple.
            The shape of input data WITHOUT the batch dimension.
            For the RNN this is (T, M), where T is the seq len and M is the vocab size.
        C: int.
            Number of classes in the dataset / vocab size.
        embedding_dim: int.
            Number of neurons in the Embedding layer (M).
        rnn_units: int.
            Number of neurons in the GRU layer (H).

        NOTE: This has the same architecture as GRU_RNN1Mini (only number of units different) so you can build this
        with one line of code :)
        '''
        pass


class GRU_RNN2(RNN):
    '''Recurrent neural network with a two GRU layers.

    Embedding → GRU → Dropout → GRU → Dropout → Dense

    Both the input and output layer have `vocab_sz` units. The output layer uses regular softmax activation.

    All layers use He/Kaiming weight initialization.
    '''
    def __init__(self, input_feats_shape, C, embedding_dim=64, rnn_units=(384, 384), dropout_rates=(0.0, 0.2)):
        '''GRU_RNN2 constructor

        Parameters:
        -----------
        input_feats_shape: tuple.
            The shape of input data WITHOUT the batch dimension.
            For the RNN this is (T, M), where T is the seq len and M is the vocab size.
        C: int.
            Number of classes in the dataset / vocab size.
        embedding_dim: int.
            Number of neurons in the Embedding layer (M).
        rnn_units: tuple of ints.
            Number of neurons in each GRU layer (H).
        dropout_rate: tuple of float
            Dropout rate to use each the Dropout layer in the net.

        TODO:
        1. Call the superclass constructor to pass along parameters that `DeepNetwork` has in common.
        2. Build out the network like usual. NOTE: you should populate the self.is_recurrent_layer list.
        '''
        pass


class GRU_RNN2XL(GRU_RNN2):
    '''Larger recurrent neural network with a two GRU layers.

    Embedding → GRU → Dropout → GRU → Dropout → Dense

    Both the input and output layer have `vocab_sz` units. The output layer uses regular softmax activation.

    All layers use He/Kaiming weight initialization.
    '''
    def __init__(self, input_feats_shape, C, embedding_dim=96, rnn_units=[512, 512], dropout_rates=(0.1, 0.2)):
        '''GRU_RNN2 constructor

        Parameters:
        -----------
        input_feats_shape: tuple.
            The shape of input data WITHOUT the batch dimension.
            For the RNN this is (T, M), where T is the seq len and M is the vocab size.
        C: int.
            Number of classes in the dataset / vocab size.
        embedding_dim: int.
            Number of neurons in the Embedding layer (M).
        rnn_units: tuple of ints.
            Number of neurons in each GRU layer (H).
        dropout_rate: tuple of float
            Dropout rate to use each the Dropout layer in the net.

        NOTE: This has the same architecture as GRU_RNN2 (only number of units / parameters different) so you can build
        this with one line of code :)
        '''
        pass


class GRU_RNN3(RNN):
    '''Recurrent neural network with a three GRU layers.

    Embedding → GRU → Dropout → GRU → Dropout → GRU → Dropout → Dense

    Both the input and output layer have `vocab_sz` units. The output layer uses regular softmax activation.

    All layers use He/Kaiming weight initialization.
    '''
    def __init__(self, input_feats_shape, C, embedding_dim=96, rnn_units=(512, 512, 512), dropout_rates=(0.2, 0.2, 0.2)):
        '''GRU_RNN3 constructor

        Parameters:
        -----------
        input_feats_shape: tuple.
            The shape of input data WITHOUT the batch dimension.
            For the RNN this is (T, M), where T is the seq len and M is the vocab size.
        C: int.
            Number of classes in the dataset / vocab size.
        embedding_dim: int.
            Number of neurons in the Embedding layer (M).
        rnn_units: tuple of ints.
            Number of neurons in each GRU layer (H).
        dropout_rate: tuple of float
            Dropout rate to use each the Dropout layer in the net.

        TODO:
        1. Call the superclass constructor to pass along parameters that `DeepNetwork` has in common.
        2. Build out the network like usual. NOTE: you should populate the self.is_recurrent_layer list.
        '''
        pass
