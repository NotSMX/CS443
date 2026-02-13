'''hebb_net.py
Bio-inspired neural network that implements the Hebbian learning rule and competition among neurons in the network
YOUR NAMES HERE
CS 443: Bio-Inspired Machine Learning
Project 1: Hebbian Learning
'''
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython import display

from viz import draw_grid_image


class HebbNet:
    '''Single layer bio-inspired neural network in which neurons compete with each other and learning occurs via a
    competitive variant of a Hebbian learning rule (Oja's Rule).

    NOTE: This network should once again be implemented in 100% TensorFlow, except where noted below.
    '''
    def __init__(self, num_features, num_neurons, k=6, inhib_value=-0.4, load_wts=False, saved_wts_path='export/wts.npy'):
        '''Hebbian network constructor

        Parameters:
        -----------
        num_features: int.
            Num input features (M).
        num_neurons: int.
            Num of neurons in the network (H).
        k: int.
            In the neural competition that occurs when processing each data sample, the neuron that achieves the kth
            highest net_in value ("neuron came in kth place") is inhibited, which means the kth place neuron gets netAct
            value of `-inhib_value`.
        inhib_value: float.
            Non-positive number (≤0) that represents the netAct value assigned to the inhibited neuron (with the kth
            highest netAct value).
        load_wts: bool.
            Whether to load weights previously saved off by the network after successful training.
        saved_wts_path: str.
            Path from the working project directory where the weights previously saved by the net are stored.
            Used if `load_wts` is True.

        TODO:
        - Create instance variables for the parameters
        - Initialize the wts
            - If loading wts, set the wts by loading the previously saved .npy wt file.
            Use `np.load` (this use of NumPy is allowed).
            - Otherwise, initialize the network wts as a tensor containing values sampled from a standard normal
            distribution (stddev = 1.0). shape=(M, H). Should NOT be a `tf.Variable` because we are not tracking
            gradients here.
        '''
        if load_wts:
            self.wts = tf.constant(np.load(saved_wts_path))
            print('Loaded stored wts.')
        else:
            # TODO: Initialize the weights
            pass
    def get_wts(self):
        '''Returns the Hebbian network wts'''
        pass

    def set_wts(self, wts):
        '''Replaces the Hebbian network weights with `wts` passed in as a parameter.

        Parameters:
        -----------
        wts: tf.float32 tensor. shape=(M, H).
            New Hebbian network weights.
        '''
        pass

    def net_in(self, x):
        '''Computes the Hebbian network Dense net_in based on the data `x`.

        Parameters:
        -----------
        x: ndarray. shape=(B, M)

        Returns:
        -----------
        tf.float32 tensor. shape=(B, H).
            netIn
        '''
        pass

    def net_act(self, net_in):
        '''Compute the Hebbian network activation, which is a function that reflects competition among the neurons
        based on their net_in values.

        NetAct (also see notebook):
        - 1 for neuron that achieves highest net_in value to sample i
        - -delta for neuron that achieves kth highest net_in value to sample i
        - 0 for all other neurons

        Parameters:
        -----------
        net_in: tf.float32 tensor. shape=(B, H).

        Returns:
        -----------
        tf.float32 tensor. shape=(B, H).
            net_act

        NOTE:
        - It may be helpful to think of this as a two step process: dealing with the winner and then kth place inhibition
        separately.
        - It may be helpful to think of competition as an assignment operation. But you may not be able to use []
        indexing. Instead, refer to your work in the competitive_hebb notebook for a TensorFlow substitute for
        arange indexing.
        - += is a valid TensorFlow operator.
        '''
        pass

    def update_wts(self, x, net_in, net_act, lr, eps=1e-10):
        '''Update the Hebbian network wts according to a modified Hebbian learning rule (competitive Oja's rule).
        After computing the weight change based on the current mini-batch, the weight changes (gradients) are normalized
        by the largest gradient (in absolute value). This has the effect of making the largest weight change equal in
        absolute magnitude to the learning rate `lr`. See notebook for equations.

        Parameters:
        -----------
        net_in: tf.float32 tensor. shape=(B, H)
        net_act: tf.float32 tensor. shape=(B, H)
        lr: float. Learning rate hyperparameter
        eps: float. Small non-negative number used in the wt normalization step to prevent possible division by 0.

        NOTE:
        - This is definitely a scenario where you should the shapes of everything to guide you through and decide on the
        appropriate operation (elementwise multiplication vs matrix multiplication).
        - The `keepdims` keyword argument may be convenient here.
        '''
        pass

    def fit(self, x, epochs=1, mini_batch_sz=500, lr=1e-2, plot_wts_live=False, fig_sz=(9, 9), n_wts_plotted=(10, 10),
            print_every=1, save_wts=True, ds_feat_shape=(32, 32, 3)):
        '''Trains the Competitive Hebbian network on the training samples `x` using unsupervised Hebbian learning
        (without classes y!).

        Parameters:
        -----------
        x: tf.float32 tensor. dtype=tf.float32. shape=(N, M).
            Data samples.
        epochs: int.
            Number of epochs to train the network.
        mini_batch_sz: int.
            Mini-batch size used when training the Hebbian network.
        lr: float.
            Learning rate used with Hebbian weight update rule
        plot_wts_live: bool.
            Whether to plot the weights and update throughout training every `print_every` epochs.
        fig_sz: tuple.
            Dimensions of the plt.figure that is used to visualize the wts (if doing this).
        n_wts_plotted: tuple.
            Grid arrangement of the weights being visualized/plotting (if doing this).
            Example: (10, 10) means wts of 100 (=10*10) neurons are shown in a 10x10 image grid.
        print_every: int.
            How often, in epochs, to print the min and max weight values across the net and draw the wt grid
            (if doing this)
        save_wts: bool.
            Whether to save the Hebbian network wts (to self.saved_wts_path) after training finishes.
        ds_feat_shape: tuple.
            Shape of the original image dataset (WITHOUT batch dimension). Used for wt visualization (if doing this).

        TODO:
        Very similar workflow to usual:
        - If plotting the wts on the current epoch, update the plot (via `draw_grid_image`) to show the current wts
        `print_every` epochs.
        - In each epoch setup mini-batch. You can sample with replacement or without replacement (shuffle) between
        epochs (your choice).
        - Compute forward pass for each mini-batch then update the weights.
        - Print out which epoch we are on `print_every` epochs
        - When training is done, save the wts if `save_wts` is True. Using `np.save` is totally fine here.
        '''
        N = len(x)

        if plot_wts_live:
            fig = plt.figure(figsize=fig_sz)

        # This is done every print_every epochs
        if plot_wts_live:
            draw_grid_image(tf.transpose(self.wts), n_wts_plotted[0], n_wts_plotted[1],
                            title=f'Net receptive fields (Epoch {e})',
                            sample_dims=ds_feat_shape)
            display.clear_output(wait=True)
            display.display(fig)
            time.sleep(0.001)
        else:
            print(f'Starting epoch {e}/{epochs}')

        # This happens at the end
        if save_wts:
            print('Saving weights...', end='')
            np.save(self.saved_wts_path, self.get_wts())
            print('Done!')
