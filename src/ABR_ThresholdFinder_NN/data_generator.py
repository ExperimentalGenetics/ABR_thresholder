"""
This script provides the data generator for the ABR-thresholder, which consists of two neural networks.
"""

import numpy as np
from tensorflow import keras


class DataGenerator(keras.utils.Sequence):
    """
    Generates data for Keras
    """

    def __init__(self, list_IDs, df, value_cols,
                 main_label_col, dim, freq_label_col=None, sl_label_col=None,
                 batch_size=64, shuffle=True,
                 sample_weight_col=[]):

        # Output layers needed to map the sample weights
        # Initialization
        if freq_label_col is None:
            freq_label_col = []
        if sl_label_col is None:
            sl_label_col = []

        self.batch_size = batch_size
        self.df = df
        self.value_cols = value_cols
        self.main_label_col = main_label_col
        self.freq_label_col = freq_label_col
        self.sl_label_col = sl_label_col
        self.dim = dim
        self.list_IDs = list_IDs
        self.indexes = np.arange(len(self.list_IDs))
        self.shuffle = shuffle
        self.sample_weight_col = sample_weight_col
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch.
        :return: int
            Number of batches.
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """
        Generates one batch of data
        """
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y, s = self.__data_generation(list_IDs_temp)

        return X, y, s

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """
        Generates data containing batch_size samples
        """
        # Initialization
        X = np.empty((self.batch_size, self.dim))
        y_main = np.empty((self.batch_size, len(self.main_label_col)), dtype=int)
        if len(self.freq_label_col) > 0:
            y_freq = np.empty((self.batch_size, len(self.freq_label_col)), dtype=int)
        if len(self.sl_label_col) > 0:
            y_sl = np.empty((self.batch_size, len(self.sl_label_col)), dtype=int)

        sample_w = np.full(self.batch_size, 1)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i, ] = self.df.loc[ID, self.value_cols].values
            X_batch = X[:, :, np.newaxis]

            # Store class
            y_main[i] = self.df.loc[ID, self.main_label_col]
            if len(self.freq_label_col) > 0:
                y_freq[i] = self.df.loc[ID, self.freq_label_col]

            if len(self.sl_label_col) > 0:
                y_sl[i] = self.df.loc[ID, self.sl_label_col]

            # Store sample weight
            if len(self.sample_weight_col) > 0:
                sample_w[i] = self.df.loc[ID, self.sample_weight_col]

        # Concatenate predictions
        y_all = {'main_prediction': y_main}
        if len(self.freq_label_col) > 0:
            y_all['frequency_prediction'] = y_freq
        if len(self.sl_label_col) > 0:
            y_all['sl_prediction'] = y_sl

        # Concatenate sample weights
        sample_w_all = {'main_prediction': sample_w}
        if len(self.freq_label_col) > 0:
            sample_w_all['frequency_prediction'] = np.full(self.batch_size, 1)
        if len(self.sl_label_col) > 0:
            sample_w_all['sl_prediction'] = np.full(self.batch_size, 1)

        return X_batch, y_all, sample_w_all
