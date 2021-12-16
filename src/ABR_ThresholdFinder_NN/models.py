"""
This script provides methods for creating and compiling the two neural networks that form the ABR-thresholder.
"""

from tensorflow import keras
from tensorflow.keras.layers import Activation

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dropout, Flatten, Dense, BatchNormalization, AveragePooling1D, \
    MaxPooling1D

from ABR_ThresholdFinder_NN.swish_activation_function import swish


def create_model_1(_freq_len, _sl_len):
    """
    Creates the first neural network.
    :param _freq_len: int
        Number of valid frequencies.
    :param _sl_len: int
        Number of valid sound levels.
    :return: Keras model
        Neural network with three output branches:
        - a main branch for predicting whether a mouse will hear for a given stimulus, a pair of frequency and sound level
        - a branch for predicting the frequency for which a given hearing curve was recorded
        - a branch for predicting the sound level for which a given hearing curve was recorded
    """
    # Main model framework
    input_shape = (1000, 1)
    inputs = Input(shape=input_shape)

    x = BatchNormalization()(inputs)

    x = Conv1D(filters=256, kernel_size=256, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)

    x = Conv1D(filters=128, kernel_size=128, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)

    x = AveragePooling1D(pool_size=4)(x)
    x = Dropout(rate=0.1)(x)

    x = Conv1D(filters=64, kernel_size=64, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)

    x = Conv1D(filters=32, kernel_size=32, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)

    x = AveragePooling1D(pool_size=2)(x)
    x = Dropout(rate=0.1)(x)

    x = Conv1D(filters=16, kernel_size=16, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)

    x = Conv1D(filters=8, kernel_size=8, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)

    x = AveragePooling1D(pool_size=2)(x)
    x = Dropout(rate=0.1)(x)

    x = Conv1D(filters=4, kernel_size=4, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)

    x = Conv1D(filters=1, kernel_size=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)

    x = Flatten()(x)
    x = Dense(32, activation=swish)(x)
    # x = Dropout(rate=0.1)(x)
    full = x

    # Output branches
    # Define output branches of model
    main_branch = Dense(1, activation='sigmoid', name='main_prediction')(full)
    out = [main_branch]
    freq_branch = Dense(_freq_len, activation='softmax', name='frequency_prediction')(full)
    out.append(freq_branch)
    sl_branch = Dense(_sl_len, activation='softmax', name='sl_prediction')(full)
    out.append(sl_branch)

    # Combine model
    model = Model(inputs=inputs, outputs=out)

    return model


def create_model_2(_input_len, _main_label_len, _freq_len):
    """
    Creates the second neural network.
    :param _input_len: int
        Dimensionality of the input vector.
    :param _main_label_len: int
        Dimensionality of the main output branch.
    :param _freq_len: int
        Dimensionality of the frequency output branch.
    :return: Keras model
        Neural network with two output branches:
        - a main branch for predicting the ABR hearing threshold for given mouse ID and stimulus frequency
        - a branch for predicting the frequency for which a given hearing curve was recorded
    """
    # Main model framework
    input_shape = (_input_len, 1)
    inputs = Input(shape=input_shape)

    x = BatchNormalization()(inputs)

    x = Conv1D(filters=128, kernel_size=6, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)

    x = Conv1D(filters=64, kernel_size=5, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)

    x = MaxPooling1D(pool_size=3)(x)
    x = Dropout(rate=0.1)(x)

    x = Conv1D(filters=32, kernel_size=4, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)

    x = Conv1D(filters=16, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)

    x = Flatten()(x)
    x = Dense(64, activation=swish)(x)
    full = x

    # Define output branches of model
    main_branch = Dense(_main_label_len, activation='sigmoid',
                        name='main_prediction')(full)
    # sigmoid; softmax if not multiple-ones-hot
    out = [main_branch]

    freq_branch = Dense(_freq_len, activation='softmax',
                        name='frequency_prediction')(full)
    out.append(freq_branch)

    # Combine model
    model = Model(inputs=inputs, outputs=out)

    return model


def compile_model_1(_model, _count_gpus):
    """
    Compiles the first neural network.
    :param _model: Keras model
    :param _count_gpus: boolean
        True if multiple GPUs are available.
    :return:
        - the compiled network that can be trained with several GPUs if necessary
        - the dictionary of loss functions for the output nodes
        - the dictionary of loss weights for the output nodes
    """
    # Compiling
    if _count_gpus > 1:
        parallel_model = keras.utils.multi_gpu_model(_model, gpus=_count_gpus)
    else:
        parallel_model = _model

    # Determine loss and loss weights
    loss = {'main_prediction': 'binary_crossentropy',
            'frequency_prediction': 'categorical_crossentropy',
            'sl_prediction': 'categorical_crossentropy'}
    loss_weights = {'main_prediction': 0.8,
                    'frequency_prediction': 0.2,
                    'sl_prediction': 0.2}

    # Compile
    parallel_model.compile(optimizer='Adam',
                           loss=loss,
                           metrics={'main_prediction': 'accuracy'},
                           loss_weights=loss_weights)
    return parallel_model, loss, loss_weights


def compile_model_2(_model, _count_gpus):
    """
    Compiles the second neural network.
    :param _model: Keras model
    :param _count_gpus: boolean
        True if multiple GPUs are available.
    :return:
        - the compiled network that can be trained with several GPUs if necessary
        - the dictionary of loss functions for the output nodes
        - the dictionary of loss weights for the output nodes
    """
    # Compiling
    if _count_gpus > 1:
        parallel_model = keras.utils.multi_gpu_model(_model, gpus=_count_gpus)
    else:
        parallel_model = _model

    # Determine loss and loss weights
    loss = {'main_prediction': 'binary_crossentropy',
            'frequency_prediction': 'categorical_crossentropy'}
    loss_weights = {'main_prediction': 1,
                    'frequency_prediction': 0.2}

    # Compile
    parallel_model.compile(optimizer='Adam',
                           loss=loss,
                           metrics={'main_prediction': 'categorical_accuracy'},
                           loss_weights=loss_weights)

    return parallel_model, loss, loss_weights
