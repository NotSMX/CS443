'''image_datasets.py
Functions to load and preprocess image datasets
Daniel Yu & Jordan Wang
CS 443: Bio-Inspired Machine Learning
Project 1: Hebbian Learning
'''
import tensorflow as tf

def get_dataset(name, norm_method='global', flatten=True, eps=1e-10, verbose=True):
    '''Main function to load the dataset `name` then preprocess and return it.

    Parameters:
    -----------
    name: str.
        Name of the requested dataset to retrieve. Supported options: 'mnist', 'cifar10'.
    norm_method: str.
        Method used to preprocess the images. Supported options: 'global', 'center', 'none'.
            - 'global' means images should be standardized using the mean and standard deviation RGB triplet computed
            globally across all images in the training set.
            - 'center' means images should be centered using the mean RGB triplet computed globally across all images in
            the training set.
    flatten: bool.
        Should we flatten out the non-batch dimensions so that the shape becomes (N, M)?
    eps: float.
        Small fudge factor to prevent potential division by 0 when standardizing.
    verbose: bool.
        When true, it is ok to print out shapes, dtypes, and other debug info. When false, nothing should print in this
        function.

    Returns:
    --------
    x_train: tf.float32 tensor. shape=(N_train, Iy, Ix, n_chans) or (N_train, M)
        Training set images
    y_train: tf.int32 tensor. shape=(N_train,)
        Training set labels
    x_test: tf tensor. shape=(N_test, Iy, Ix, n_chans) or (N_test, M)
        Test set images
    y_test: tf.int32 tensor. shape=(N_test,)
        Test set labels

    NOTE:
    1. You should rely on the TensorFlow Keras built-in datasets module to acquire the datasets.
    2. Min-max normalize the images features to floats between 0-1 before performing any addition preprocessing.
    3. This function (and the file more generally) should be written in TensorFlow. You should not import NumPy.
    '''
    name = name.lower()

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------
    if name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        n_chans = 3

    elif name == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # Add singleton channel dimension: (N, 28, 28) -> (N, 28, 28, 1)
        x_train = tf.expand_dims(x_train, axis=-1)
        x_test  = tf.expand_dims(x_test, axis=-1)
        n_chans = 1

    else:
        raise ValueError(f"Unsupported dataset: {name}")

    # Labels come as (N, 1) for CIFAR-10; ensure (N,)
    y_train = tf.squeeze(y_train)
    y_test  = tf.squeeze(y_test)

    # --------------------------------------------------
    # Convert to float and scale to [0, 1]
    # --------------------------------------------------
    x_train = tf.cast(x_train, tf.float32) / 255.0
    x_test  = tf.cast(x_test, tf.float32) / 255.0

    # --------------------------------------------------
    # Global preprocessing (computed on training set)
    # --------------------------------------------------
    if norm_method in ['global', 'center']:
        mean = tf.reduce_mean(x_train, axis=[0, 1, 2])

        if norm_method == 'global':
            std = tf.math.reduce_std(x_train, axis=[0, 1, 2])
            x_train = (x_train - mean) / (std + eps)
            x_test  = (x_test  - mean) / (std + eps)

        elif norm_method == 'center':
            x_train = x_train - mean
            x_test  = x_test  - mean

    elif norm_method == 'none':
        pass

    else:
        raise ValueError(f"Unsupported norm_method: {norm_method}")

    # --------------------------------------------------
    # Optional flattening
    # --------------------------------------------------
    if flatten:
        x_train = tf.reshape(x_train, [tf.shape(x_train)[0], -1])
        x_test  = tf.reshape(x_test,  [tf.shape(x_test)[0], -1])

    # --------------------------------------------------
    # Verbose output
    # --------------------------------------------------
    if verbose:
        print(f"Dataset: {name}")
        print("x_train:", x_train.shape, x_train.dtype)
        print("y_train:", y_train.shape, y_train.dtype)
        print("x_test: ", x_test.shape, x_test.dtype)
        print("y_test: ", y_test.shape, y_test.dtype)

    return x_train, y_train, x_test, y_test


def train_val_split(x_train, y_train, prop_val=0.1):
    '''Subdivides the provided training set into a (smaller) training set and a validation set, composed of the last
    `prop_val` proportion of samples in x_train/y_train.

    Parameters:
    -----------
    x_train: tf.float32 tensor. shape=(N, Iy, Ix, n_chans) or (N, M)
        The original training set data
    y_train: tf.int32 tensor. shape=(N,)
        The original training set labels
    prop_val: float.
        Proportion of the original training set to reserve for the validation set.


    Returns:
    --------
    x_train: tf.float32 tensor. shape=(N_train_new, Iy, Ix, n_chans) or (N_train_new, M)
        Training set images
    y_train: tf.int32 tensor. shape=(N_train_new,)
        Training set labels
    x_val: tf tensor. shape=(N_val, Iy, Ix, n_chans) or (N_val, M)
        Validation set images
    y_val: tf.int32 tensor. shape=(N_val,)
        Validation set labels
    '''
    # Total number of samples
    N = tf.shape(x_train)[0]

    # Number of validation samples
    n_val = tf.cast(tf.cast(N, tf.float32) * prop_val, tf.int32)

    # Split index
    split_idx = N - n_val

    # Perform split
    x_train_new = x_train[:split_idx]
    y_train_new = y_train[:split_idx]

    x_val = x_train[split_idx:]
    y_val = y_train[split_idx:]

    return x_train_new, y_train_new, x_val, y_val


def preprocess_nonlinear(x, n=4.0):
    '''Preprocessor for the nonlinear decoder input data. Applies the ReLU function raised to the `n` power.

    (Week 3)

    Parameters:
    -----------
    x: tf.float32 tensor. shape=(N, M)
        Hebbian network net_in values.
    n: float.
        Power to raise the output of the ReLU applied to `x`.

    Returns:
    --------
    tf.float32 tensor. shape=(N, M).
        Data transformed by ReLU raised to the `n` power.
    '''
    pass


def occlude_images(x, region='top', image_dims=(28, 28, 1)):
    '''Occludes/deletes the content in half of each image passed in.

    (This function is provided / should not required modification)

    Parameters:
    -----------
    x: tf.float32 tensor. shape=(N, M)
        Flatten image data.
    region: str.
        Region in each image to occlude. Supported options: 'top', 'bottom'
            - 'top' means occlude top half of images
            - 'bottom' means occlude bottom half of images
    image_dims: tuple of ints.
        The original unflattened shape of the image data without the batch dimension.

    Returns:
    --------
    x_flat: tf.float32 tensor. shape=(N, M).
        Occluded images.
    mask_flat: tf.float32 tensor. shape=(N, M).
        Boolean mask specifying whether occlusion was applied to each pixel each image.
    '''
    N, M = x.shape

    # Reshape to 2D
    x_2d = tf.reshape(x, [N, image_dims[0], image_dims[1], image_dims[2]])

    half_ind = image_dims[0] // 2

    if region == 'top':
        # Make mask to 0 out the top half of each image
        occlusion_mask = tf.concat([tf.zeros([N, half_ind, image_dims[1], image_dims[2]]),
                                    tf.ones([N, half_ind, image_dims[1], image_dims[2]])], axis=1)
    elif region == 'bottom':
        # Make mask to 0 out the bottom half of each image
        occlusion_mask = tf.concat([tf.ones([N, half_ind, image_dims[1], image_dims[2]]),
                                    tf.zeros([N, half_ind, image_dims[1], image_dims[2]])], axis=1)

    # Apply the mask: Where mask = 1, keep pixels. Where 0.0, set to min img value
    x_min = tf.reduce_min(x)
    x_occluded = tf.where(tf.cast(occlusion_mask, tf.bool), x_2d, x_min)

    # Flatten occluded images and mask back to (N, M)
    x_flat = tf.reshape(x_occluded, [N, M])
    mask_flat = tf.reshape(occlusion_mask, [N, M])
    return x_flat, mask_flat
