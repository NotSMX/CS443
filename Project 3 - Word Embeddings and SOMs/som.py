'''som.py
Self-organizing map implemented in NumPy
DANIEL YU & JORDAN WANG
CS 443: Bio-Inspired Machine Learning
Project 3: Word Embeddings and Self-Organizing Maps (SOMs)
'''
import numpy as np

def lin2sub(ind, the_shape):
    '''Utility function that takes a linear index and converts it to subscripts.
    No changes necessary here.

    Parameters:
    ----------
    ind: int. Linear index to convert to subscripts
    the_shape: tuple. Shape of the ndarray from which `ind` was taken.

    Returns:
    ----------
    tuple of subscripts

    Example: ind=2, the_shape=(2,2) -> return (1, 0).
        i.e. [[_, _], [->SUBSCRIPT OF THIS ELEMENT<-, _]]
    '''
    return np.unravel_index(ind, the_shape)


class SOM:
    '''A 2D self-organzing map (SOM) neural network.
    '''
    def __init__(self, map_sz, num_feats, feature_range=(0, 1), seed=0):
        '''Creates a new SOM with random weights in range [0, 1]

        Parameters:
        ----------
        map_sz: tuple of 2 ints. (n_rows, n_cols).
            Number of units in each dimension of the SOM.
            e.g. map_sz=(9, 10) -> the SOM will have 9x10 units arranged in a grid of 9 rows and 10 columns.
        num_feats: int.
            Number of features in a SINGLE data sample feature vector.
        feature_range: tuple of int. format: (min_feature_value, max_feature_value).
            Range of features values. Initialize weights within the same range.
        seed: int.
            Random number generator seed to use when initializing the weights.

        TODO: Initialize weights (self.wts) to uniform random values within the feature range.
            shape=(n_rows, n_cols, num_feats).
        '''
        self.num_feats = num_feats
        self.n_rows, self.n_cols = map_sz
 
        rng = np.random.default_rng(seed)
        low, high = feature_range
        self.wts = rng.uniform(low, high, size=(self.n_rows, self.n_cols, num_feats))
 
        nr, nc = np.meshgrid(np.arange(self.n_rows), np.arange(self.n_cols), indexing='ij')
        self.som_grid_rows = nr
        self.som_grid_cols = nc

    def get_wts(self):
        '''Returns the weight vector.

        No changes necessary here.
        '''
        return self.wts

    def get_bmu(self, input_vector):
        '''Compute the best matching unit (BMU) given an input data vector. THE BMU is the unit with
        the closest weights to the data vector. Uses Euclidean distance (L2 norm) as the distance
        metric.

        Parameters:
        ----------
        input_vector: ndarray. shape=(num_feats,). One data sample vector.

        Returns:
        ----------
        tuple of the BMU (row, col) in the SOM grid.

        NOTE: For efficiency, you may not use any loops.
        '''
        diff = self.wts - input_vector          # (n_rows, n_cols, num_feats)
        dist = np.sqrt(np.sum(diff ** 2, axis=2))  # (n_rows, n_cols)
        bmu_lin = np.argmin(dist)
        bmu_rc = lin2sub(bmu_lin, (self.n_rows, self.n_cols))
        return (int(bmu_rc[0]), int(bmu_rc[1]))

    def get_nearest_wts(self, data):
        '''Find the nearest SOM wt vector to each of data sample vectors.

        Parameters:
        ----------
        data: ndarray. shape=(N, num_feats) for N data samples.

        Returns:
        ----------
        ndarray. shape=(N, num_feats). The most similar weight vector for each data sample vector.

        NOTE: A loop is fine here.
        '''
        nearest = np.zeros((len(data), self.num_feats))
        for i, x in enumerate(data):
            bmu_rc = self.get_bmu(x)
            nearest[i] = self.wts[bmu_rc]
        return nearest

    def gaussian(self, bmu_rc, sigma):
        '''Generates a "normalized" 2D Gaussian, where the max value is 1, and is centered on `bmu_rc`.

        Parameters:
        ----------
        bmu_rc: tuple. (row, col) in the SOM grid of current best-matching unit (BMU).
        sigma: float. Standard deviation of the Gaussian at the current training iteration.
            The parameter passed in has already been decayed.

        Returns:
        ----------
        ndarray. shape=(n_rows, n_cols). 2D Gaussian, weighted by the the current learning rate.

        TODO:
        - Evaluate a Gaussian on a 2D grid with shape=(n_rows, n_cols) centered on `bmu_rc`.


        HINT:
        This will likely involve generating 2D GRIDS of (row, col) index values (i.e. positions in
        the 2D grid) in the range [0, ..., n_rows-1, 0, ..., n_cols-1].
            shape of som_grid_cols: (n_rows, n_cols)
            shape of som_grid_rows: (n_rows, n_cols)
        You already solved this problem in Project 0 of CS343 :)
        If you adopt this approach, see the initialization of these row and column matrices in the
        constructor and use them.

        NOTE: For efficiency, you should not use any for loops.
        '''
        bmu_r, bmu_c = bmu_rc
        dist_sq = (self.som_grid_rows - bmu_r) ** 2 + (self.som_grid_cols - bmu_c) ** 2
        return np.exp(-dist_sq / (2 * sigma ** 2))

    def update_wts(self, input_vector, bmu_rc, lr, sigma):
        '''Applies the SOM update rule to change the BMU (and neighboring units') weights,
        bringing them all closer to the data vector (cooperative learning).

        Parameters:
        ----------
        input_vector: ndarray. shape=(num_feats,). One data sample.
        bmu_rc: tuple. BMU (x,y) position in the SOM grid.
        lr: float. Current learning rate during learning.
        sigma: float. Current standard deviation of Gaussian neighborhood in which units cooperate
            during learning.

        NOTE: For efficiency, you should not use any loops.
        '''
        gauss = self.gaussian(bmu_rc, sigma)[:, :, np.newaxis]
        input_vec = input_vector[np.newaxis, np.newaxis, :]

        self.wts = (self.wts + lr * gauss * (input_vec - self.wts)).astype(np.float32)

    def decay_param(self, curr_iter, num_iters, initial_val, final_val):
        '''Takes a hyperparameter (e.g. lr, sigma) and applies a exponential time-dependent decay function.

        Parameters:
        ----------
        curr_iter: int.
            The current training iteration number (1st iter is 0).
        num_iter: int.
            The total number of training iterations.
        initial_val: float.
            Initial (not current) value of a hyperparameter (e.g. lr, sigma) whose value we will decay.
        final_val: float.
            The desired final value for the hyperparameter (e.g. lr, sigma) at the end of training.

        Returns:
        ----------
        float.
            The decayed parameter.
        '''
        tau = (num_iters - 1) / np.log(initial_val / final_val)
        return initial_val * np.exp(-curr_iter / tau)

    def fit(self, x, epochs, lr_initial=0.2, lr_final=0.01, sigma_initial=0.2, sigma_final=0.01, print_every=1,
            seed=0, verbose=True):
        '''Train the SOM on data

        Parameters:
        ----------
        x: ndarray. shape=(N, num_feats) N training data samples.
        n_epochs: int. Number of training epochs to do
        lr: float. INITIAL learning rate during learning. This will decay with time
            (iteration number). The effective learning rate will only equal this if t=0.
        lr_decay: float. Multiplicative decay rate for learning rate.
        sigma: float. INITIAL standard deviation of Gaussian neighborhood in which units
            cooperate during learning. This will decay with time (iteration number).
            The effective learning rate will only equal this if t=0.
        sigma_decay: float. Multiplicative decay rate for Gaussian neighborhood sigma.
        print_every: int. Print the epoch, lr, sigma, and BMU error every `print_every` epochs.
            NOTE: When first implementing this, ignore "BMU error". You will be computing this soon,
            at which point you can go back and add it.
        verbose: boolean. Whether to print out debug information at various stages of the algorithm.
            NOTE: if verbose=False, nothing should print out during training. Messages indicating start and
            end of training are fine.

        TODO:
        Although this is an unsupervised learning algorithm, the training process is similar to usual:
        - Train SGD-style — one sample at a time. For each epoch you should either sample with replacement or without
        replacement (shuffle) between epochs (your choice).
            - If you shuffle the entire dataset each epoch, be careful not to accidentally modify the original data
            passed in!
        - Within each epoch: compute the BMU of each data sample, update its weights and those of its neighbors, and
        decay the learning rate and Gaussian neighborhood sigma.
        '''
        N = len(x)
        rng = np.random.default_rng(seed)
        num_iters = epochs * N
 
        print(f'Starting training for {epochs} epochs...')
 
        t = 0
        for e in range(epochs):
            inds = rng.permutation(N)
            x_shuf = x[inds]
 
            for i in range(N):
                lr = self.decay_param(t, num_iters, lr_initial, lr_final)
                sigma = self.decay_param(t, num_iters, sigma_initial, sigma_final)
                bmu_rc = self.get_bmu(x_shuf[i])
                self.update_wts(x_shuf[i], bmu_rc, lr, sigma)
                t += 1
 
            if verbose and (e + 1) % print_every == 0:
                bmu_err = self.error(x)
                print(f'Epoch {e+1}/{epochs} | lr={lr:.5f} | sigma={sigma:.5f} | BMU error={bmu_err:.5f}')
 
        print(f'Finished training.')

    def error(self, data):
        '''Computes the quantization error: average error incurred by approximating all data vectors
        with the weight vector of the BMU.

        Parameters:
        ----------
        data: ndarray. shape=(N, num_feats) for N data samples.

        Returns:
        ----------
        float. Average error over N data vectors

        TODO:
        - Progressively average the Euclidean distance between each data vector
        and the BMU weight vector.
        '''
        nearest = self.get_nearest_wts(data)
        return float(np.mean(np.sqrt(np.sum((data - nearest) ** 2, axis=1))))

    def u_matrix(self):
        '''Compute U-matrix, the distance each SOM unit wt and that of its 8 local neighbors.

        Returns:
        ----------
        ndarray. shape=(map_sz, map_sz). Total Euclidan distance between each SOM unit
            and its 8 neighbors.

        NOTE:
        - Remember to normalize the U-matrix so that its range of values always span [0, 1].
        - Loops are fine here.

        '''
        u = np.zeros((self.n_rows, self.n_cols))
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                total, count = 0.0, 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr2, nc2 = r + dr, c + dc
                        if 0 <= nr2 < self.n_rows and 0 <= nc2 < self.n_cols:
                            total += np.sqrt(np.sum((self.wts[r, c] - self.wts[nr2, nc2]) ** 2))
                            count += 1
                u[r, c] = total / count if count else 0.0
        u_min, u_max = u.min(), u.max()
        if u_max > u_min:
            u = (u - u_min) / (u_max - u_min)
        return u
