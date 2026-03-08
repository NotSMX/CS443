'''softmax_layer.py
Constructs, trains, tests single layer neural network with softmax activation function.
Olurinfe Onabolu
CS 343: Neural Networks
Spring 2025
Project 2: Multilayer Perceptrons
'''
import numpy as np
from optimizer import Adam


class SoftmaxLayer:
    '''SoftmaxLayer is a class for single layer networks with softmax activation and cross-entropy loss
    in the output layer.
    '''
    def __init__(self, num_output_units):
        '''SoftmaxLayer constructor

        Parameters:
        -----------
        num_output_units: int. Num output units. Equal to # data classes.
        '''
        # Network weights
        self.wts = None
        # Bias
        self.b = None
        # Number of data classes C
        self.num_output_units = num_output_units

    def accuracy(self, y, y_pred):
        '''Computes the accuracy of classified samples. Proportion correct

        Parameters:
        -----------
        y: ndarray. int-coded true classes. shape=(Num samps,)
        y_pred: ndarray. int-coded predicted classes by the network. shape=(Num samps,)

        Returns:
        -----------
        float. accuracy in range [0, 1]
        '''
        acc = np.mean(y == y_pred)
        return acc

    def net_in(self, features):
        '''Computes the net input (net weighted sum)

        Parameters:
        -----------
        features: ndarray. input data. shape=(num images (in mini-batch), num features)
        i.e. shape=(N, M)

        Note: shape of self.wts = (M, C), for C output neurons

        Returns:
        -----------
        net_input: ndarray. shape=(N, C)
        '''
        net_input = np.dot(features, self.wts) + self.b
        return net_input

    def one_hot(self, y, num_classes):
        '''One-hot codes the output classes for a mini-batch

        Parameters:
        -----------
        y: ndarray. int-coded class assignments of training mini-batch. 0,...,C-1
        num_classes: int. Number of unique output classes total

        Returns:
        -----------
        y_one_hot: One-hot coded class assignments.
            e.g. if y = [0, 2, 1] and num_classes (C) = 4 we have:
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]]
        '''
        y_one_hot = np.eye(num_classes)[y]
        return y_one_hot

    def fit(self, features, y, n_epochs=100, lr=0.0001, mini_batch_sz=256, reg=0, r_seed=None, verbose=2, x_val=None, y_val=None, val_freq=50):
        '''Trains the network to data in `features` belonging to the int-coded classes `y`.
        Implements stochastic mini-batch gradient descent

        Parameters:
        -----------
        features: ndarray. shape=(Num samples N, num features M)
        y: ndarray. int-coded class assignments of training samples. 0,...,numClasses-1
        n_epochs: int. Number of training epochs
        lr: float. Learning rate
        mini_batch_sz: int. Batch size per training iteration.
            i.e. Chunk this many data samples together to process with the model on each training
            iteration. Then we do gradient descent and update the wts. NOT the same thing as an epoch.
        reg: float. Regularization strength used when computing the loss and gradient.
        r_seed: None or int. Random seed for weight and bias initialization.
        verbose: int. 0 means no print outs. Any value > 0 prints epoch-level training info.
        x_val: ndarray or None. Validation features. shape=(Num val samples, num features)
        y_val: ndarray or None. Validation labels (int-coded)
        val_freq: int. Validation check frequency in epochs.

        Returns:
        -----------
        train_loss_history: Python list of floats. Recorded average training loss per epoch.
        val_loss_history: Python list of floats or None. Validation losses at checkpoints.

                NOTE:
                Validation metrics are always printed on the first and last epoch when validation
                data is provided, regardless of val_freq.
        '''
        
        np.random.seed(r_seed)
        num_features = features.shape[1]
        self.wts = np.random.normal(0, 0.01, (num_features, self.num_output_units))
        self.b = np.random.normal(0, 0.01, (self.num_output_units,))

        N = features.shape[0]
        iterations_per_epoch = max(int(np.ceil(N / mini_batch_sz)), 1)
        train_loss_history = []
        val_loss_history = [] if (x_val is not None and y_val is not None) else None

        w_opt = Adam(lr=lr)
        b_opt = Adam(lr=lr)

        for epoch in range(1, n_epochs + 1):
            batch_losses = []

            for _ in range(iterations_per_epoch):
                batch_indices = np.random.choice(N, mini_batch_sz, replace=True)
                X_batch = features[batch_indices]
                y_batch = y[batch_indices]

                if mini_batch_sz == 1:
                    X_batch = X_batch.reshape(1, -1)
                    y_batch = np.array([y_batch])

                y_one_hot = self.one_hot(y_batch, self.num_output_units)
                net_act = self.activation(self.net_in(X_batch))
                loss = self.loss(net_act, y_batch, reg=reg)
                batch_losses.append(loss)

                grad_wts, grad_b = self.gradient(X_batch, net_act, y_one_hot, reg=reg)
                w_opt.prepare(self.wts, grad_wts)
                self.wts = w_opt.update_weights()
                b_opt.prepare(self.b, grad_b)
                self.b = b_opt.update_weights()

            epoch_loss = float(np.mean(batch_losses))
            train_loss_history.append(epoch_loss)

            has_val = x_val is not None and y_val is not None
            should_eval_val = False
            if has_val:
                should_eval_val = (epoch == 1) or (epoch == n_epochs) or (epoch % max(1, val_freq) == 0)

            if verbose > 0:
                print(f"Epoch {epoch}/{n_epochs} - train_loss: {epoch_loss:.4f}")

            if has_val and should_eval_val:
                val_net_act = self.activation(self.net_in(x_val))
                val_loss = float(self.loss(val_net_act, y_val, reg=reg))
                val_loss_history.append(val_loss)
                if verbose > 0:
                    val_acc = self.accuracy(y_val, self.predict(x_val))
                    print(f"Epoch {epoch}/{n_epochs} - val_loss: {val_loss:.4f}, val_acc: {100*val_acc:.2f}%")

        return train_loss_history, val_loss_history

    def predict(self, features):
        '''Predicts the int-coded class value for network inputs ('features').

        Parameters:
        -----------
        features: ndarray. shape=(mini-batch size, num features)

        Returns:
        -----------
        y_pred: ndarray. shape=(mini-batch size,).
            This is the int-coded predicted class values for the inputs passed in.
            Note: You can figure out the predicted class assignments from net_in (i.e. you dont
            need to apply the net activation function — it will not affect the most active neuron).
        '''
        net_in = self.net_in(features)
        y_pred =  np.argmax(net_in, axis=1)
        return y_pred

    def activation(self, net_in):
        '''Applies the softmax activation function on the net_in.

        Parameters:
        -----------
        net_in: ndarray. net in. shape=(mini-batch size, num output neurons)
        i.e. shape=(N, C)

        Returns:
        -----------
        f_z: ndarray. net_act transformed by softmax function. shape=(N, C)

        Tips:
        -----------
        - Remember the adjust-by-the-max trick (for each input samp) to prevent numeric overflow!
        This will make the max net_in value for a given input 0.
        - np.sum and np.max have a keepdims optional parameter that might be useful for avoiding
        going from shape=(X, Y) -> (X,). keepdims ensures the result has shape (X, 1).
        '''
        net_in -= np.max(net_in, axis=1, keepdims=True)
        exp = np.exp(net_in)
        f_z = exp / np.sum(exp, axis=1, keepdims=True)
        return f_z

    def loss(self, net_act, y, reg=0):
        '''Computes the cross-entropy loss

        Parameters:
        -----------
        net_act: ndarray. softmax net activation. shape=(mini-batch size, num output neurons)
        i.e. shape=(N, C)
        y: ndarray. correct class values, int-coded. shape=(mini-batch size,)
        reg: float. Regularization strength

        Returns:
        -----------
        loss: float. Regularized (!!!!) average loss over the mini batch

        Tips:
        -----------
        - Remember that the loss is the negative of the average softmax activation values of neurons
        coding the correct classes only.
        - It is handy to use arange indexing to select only the net_act values coded by the correct
          output neurons.
        - NO FOR LOOPS!
        - Remember to add on the regularization term, which has a 1/2 in front of it.
        '''
        
        loss = -np.mean(np.log(net_act[np.arange(y.shape[0]), y]))
        loss += 0.5 * reg * np.sum(self.wts ** 2)

        return loss

    def gradient(self, features, net_act, y, reg=0):
        '''Computes the gradient of the softmax version of the net

        Parameters:
        -----------
        features: ndarray. net inputs. shape=(mini-batch-size, Num features)
        net_act: ndarray. net outputs. shape=(mini-batch-size, C)
            In the softmax network, net_act for each input has the interpretation that
            it is a probability that the input belongs to each of the C output classes.
        y: ndarray. one-hot coded class labels. shape=(mini-batch-size, Num output neurons)
        reg: float. regularization strength.

        Returns:
        -----------
        grad_wts: ndarray. Weight gradient. shape=(Num features, C)
        grad_b: ndarray. Bias gradient. shape=(C,)

        NOTE:
        - Gradient is the same as ADALINE, except we average over mini-batch in both wts and bias.
        - NO FOR LOOPS!
        - Don't forget regularization!!!! (Weights only, not for bias)
        '''
        grad_wts = np.dot(features.T, net_act - y) / features.shape[0] + reg * self.wts
        grad_b = np.mean(net_act - y, axis=0)

        return grad_wts, grad_b

    def test_loss(self, wts, b, features, labels):
        ''' Tester method for net_in and loss
        '''
        self.wts = wts
        self.b = b

        net_in = self.net_in(features)
        print(f'net in shape={net_in.shape}, min={net_in.min()}, max={net_in.max()}')
        print('Should be\nnet in shape=(15, 10), min=0.584664799299611, max=1.411396670099296\n')

        net_act = self.activation(net_in)
        print(f'net act shape={net_act.shape}, min={net_act.min()}, max={net_act.max()}')
        print('Should be\nnet act shape=(15, 10), min=0.0665134672262976, max=0.1439281981621258\n')
        return self.loss(net_act, labels, 0), self.loss(net_act, labels, 0.5)

    def test_gradient(self, wts, b, features, labels, num_unique_classes, reg=0):
        ''' Tester method for gradient
        '''
        self.wts = wts
        self.b = b

        net_in = self.net_in(features)
        print(f'net in: {net_in.shape}, {net_in.min()}, {net_in.max()}')
        print(f'net in 1st few values of 1st input are:\n{net_in[0, :5]}\nand should be')
        print('[0.798 1.095 0.969 0.9   0.958]')

        net_act = self.activation(net_in)
        print(f'net act 1st few values of 1st input are:\n{net_act[0, :5]}\nand should be')
        print('[0.078 0.105 0.092 0.086 0.091]')

        labels_one_hot = self.one_hot(labels, num_unique_classes)
        print(f'y one hot: {labels_one_hot.shape}, sum is {np.sum(labels_one_hot)}.')
        print('You should know what the sum should be :)')

        return self.gradient(features, net_act, labels_one_hot, reg=reg)
