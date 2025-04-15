"""
This file defines a scikit-learn compatible DEVNET anomaly detection model.
It supports two MLP architectures:
  - 'shallow': one hidden layer with 20 units (default).
  - 'deep': three hidden layers with sizes 1000, 250, and 20.

The model is compiled with the RMSprop optimizer and a deviation loss.
Training is performed for 50 epochs with 20 mini-batches per epoch.
"""

import numpy as np
from sklearn.base import BaseEstimator
from keras.optimizers import RMSprop
from devnet import (
    deviation_loss,
    dev_network_s,
    dev_network_d,
    batch_generator_sup
)

class CustomDevNet(BaseEstimator):
    def __init__(self, architecture='shallow', batch_size=512, nb_batch=20, epochs=50, 
                 random_seed=42, data_format=0):
        """
        Parameters:
            architecture: 'shallow' for one hidden layer with 20 units (default) or 'deep' for three hidden layers.
            batch_size: Batch size for training.
            nb_batch: Number of mini-batches per epoch.
            epochs: Number of training epochs.
            random_seed: Random seed for reproducibility.
            data_format: 0 for dense data, 1 for sparse.
        """
        self.architecture = architecture
        self.batch_size = batch_size
        self.nb_batch = nb_batch
        self.epochs = epochs
        self.random_seed = random_seed
        self.data_format = data_format

        self.model_ = None  
        self.input_shape_ = None

    def fit(self, X, y):
        self.input_shape_ = X.shape[1:]
        # Identify anomalies (non-zero) vs normals.
        outlier_indices = np.where(y != 0)[0]
        inlier_indices = np.where(y == 0)[0]
        print("Training data: {} samples, {} outliers, {} inliers.".format(
            X.shape[0], len(outlier_indices), len(inlier_indices)))
        
        # Select network architecture.
        if self.architecture == 'deep':
            self.model_ = dev_network_d(self.input_shape_)
        else:
            self.model_ = dev_network_s(self.input_shape_)
        
        # Compile the model with the RMSprop optimizer and deviation loss.
        optimizer = RMSprop(clipnorm=1.)
        self.model_.compile(loss=deviation_loss, optimizer=optimizer)
        
        print(self.model_.summary())
        
        rng = np.random.RandomState(self.random_seed)
        generator = batch_generator_sup(
            X, outlier_indices, inlier_indices,
            batch_size=self.batch_size, nb_batch=self.nb_batch, rng=rng)
        
        self.model_.fit(generator, steps_per_epoch=self.nb_batch, epochs=self.epochs, verbose=1)
        return self

    def predict(self, X):
        if self.model_ is None:
            raise ValueError("Model is not trained yet.")
        if self.data_format == 0:
            scores = self.model_.predict(X)
        else:
            data_size = X.shape[0]
            scores = np.zeros((data_size, 1))
            i, count = 0, self.batch_size
            while i < data_size:
                subset = X[i:count].toarray()
                scores[i:count] = self.model_.predict(subset)
                i = count
                count += self.batch_size
                if count > data_size:
                    count = data_size
        return scores