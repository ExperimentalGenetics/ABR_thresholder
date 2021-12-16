"""
This script provides methods for preparing the data to which the ABR-thresholder is applied. The data consists of ABR
hearing curves. The ABR-thresholder consists of two neural networks.
"""

import os
import re
import math
import warnings

import pandas as pd
import numpy as np

# Tensorflow and tf.keras
from tensorflow import keras


def weight_sample4model_1(_data):
    """
    Creates sample weights for the first neural network of the ABR-thresholder.
    Some hearing curves, our samples, have been evaluated several times and
    the assigned hearing thresholds have been validated where appropriate.
    The validated samples are higher weighted during the training.
    :param _data: pandas-data-frame
        Contains ABR hearing curves plus metadata.
        Each row contains:
        - an identifier of the mouse given by the 'mouse_id'-column
        - the stimulus sound level given by the 'sound_level'-column
        - the stimulus frequency given by the 'frequency'-column
        - the ABR hearing curve recorded for a mouse at given sound_level and frequency
        - if existing, the 'validated'-column indicates whether the assigned hearing threshold has been validated by
        several experts or not.
    :return:
        - a pandas-data-frame that contains an additional column with the calculated sample weights
        - the name of the column in which the sample weights are stored
    """

    # Sum of the weights should be the sum of the classifications
    classifs = _data[['mouse_id', 'frequency']].drop_duplicates().shape[0]
    # Sum of weights for each group (validated and non-validated group)
    sum_weight_valid = round(classifs / 2)
    print('sum_weight_valid = %d' % sum_weight_valid)
    # Sum for each frequency
    sum_weight_freq = round(sum_weight_valid / len(_data['frequency'].unique()))
    print('sum_weight_freq = %d' % sum_weight_freq)
    # Data-frame with all possible combinations
    df_sweights = _data[['mouse_id', 'validated', 'frequency']].drop_duplicates().groupby(
        ['validated', 'frequency']).size()
    df_sweights = pd.DataFrame(df_sweights).reset_index()
    df_sweights.columns = ['validated', 'frequency', 'count']
    df_sweights['sweight'] = round(sum_weight_freq / df_sweights['count'], 2)
    df_sweights.loc[df_sweights['validated'] == False, 'sweight'] = round(sum_weight_freq / df_sweights['count'], 2) * 2
    df_sweights.reset_index(drop=True, inplace=True)

    # Merge the sample weights with the overall data-frame
    data1 = pd.merge(_data, df_sweights[['validated', 'frequency', 'sweight']], on=['validated', 'frequency'],
                     how='left')
    sample_weight_col = ['sweight']
    print(df_sweights)

    return data1, sample_weight_col


def weight_sample4model_2(_data):
    """
    Creates sample weights for the second neural network of the ABR-thresholder.
    Some hearing curves, our samples, have been evaluated several times and
    the assigned hearing thresholds have been validated where appropriate.
    The validated samples are higher weighted during the training.
    :param _data: pandas-data-frame
        Contains the sound level hearing predictions of the first neural network for a given mouse ID and
        a given stimulus, a pair of frequency and sound level.
    :return:
        - a pandas-data-frame that contains an additional column with the calculated sample weights
        - the name of the column in which the sample weights are stored.
    """

    # Sum of the weights should be the sum of the classifications
    classifs = _data[['mouse_id', 'frequency']].drop_duplicates().shape[0]

    # Sum of weights for each group (validated and non-validated group)
    sum_weight_valid = round(classifs / 2)
    print('sum_weight_valid = %d' % sum_weight_valid)

    # Sum for each frequency
    sum_weight_freq = round(sum_weight_valid / len(_data['frequency'].unique()))
    print('sum_weight_freq = %d' % sum_weight_freq)

    # Data-frame with all possible combinations
    df_sweights = _data[['mouse_id', 'validated', 'frequency']].drop_duplicates().groupby(
        ['validated', 'frequency']).size()
    df_sweights = pd.DataFrame(df_sweights).reset_index()
    df_sweights.columns = ['validated', 'frequency', 'count']
    df_sweights['sweight'] = round(sum_weight_freq / df_sweights['count'], 2)
    df_sweights.reset_index(drop=True, inplace=True)

    # Merge the sample weights with the overall data-frame
    data1 = pd.merge(_data, df_sweights[['validated', 'frequency', 'sweight']], on=['validated', 'frequency'],
                     how='left')
    sample_weight_col = ['sweight']
    print(df_sweights)

    return data1, sample_weight_col


def preprocess_data4model_2(_data, _sound_levels, _aul_sound_level):
    """
    Preprocess the output data of the first neural network for feeding into
    the second neural network of the ABR-thresholder.
    :param _data: pandas-data-frame
        A data frame that contains the sound level hearing predictions of the first neural network for a given mouse ID
        and a given stimulus, a pair of frequency and sound level.
    :param _sound_levels: list
        List of valid sound levels.
    :param _aul_sound_level: int
        The value to be assigned when the manually determined hearing threshold is missing.
    :return: pandas-data-frame
        A data frame with the hearing predictions of the first neural network given column-wise.
    """

    # Create new dataframe for further formatting
    data = _data.copy()

    # Raise error if sound levels of input data are not all in model sound levels
    if any(elem not in _sound_levels for elem in list(_data['sound_level'].unique())):
        raise ValueError('Input data has more sound levels than the model')

    # For each frequency map the 'AUL threshold' to an universally valid sound level (aul_sound_level) defined above
    for freq in data['frequency'].unique():
        # Determine the threshold which does not appear in the sound levels
        thresholds = data.loc[data['frequency'] == freq, 'threshold'].unique()
        soundlvls = data.loc[data['frequency'] == freq, 'sound_level'].unique()
        freq_aul_sl = [x for x in thresholds if x not in soundlvls]
        if len(freq_aul_sl) == 1:
            freq_aul_sl = freq_aul_sl[0]
        elif len(freq_aul_sl) == 0:
            freq_aul_sl = max(soundlvls) + 5
            warnings.warn('There was no AUL sound level found!')
        else:
            raise ValueError('Possibly multiple AUL sound level thresholds')
        # Change this frequency specific aul sound level to the universally valid one
        data.loc[(data['frequency'] == freq) & (data['threshold'] == freq_aul_sl), 'threshold'] = _aul_sound_level

    # Long to wide format for sound level column
    if 'validated' in data.columns:
        data1 = pd.pivot_table(data, values='predicted_hears_exact',
                               index=['mouse_id', 'mouse_group', 'sweight', 'validated', 'frequency', 'threshold'],
                               columns=['sound_level'],
                               aggfunc=np.sum).reset_index().rename_axis(None, axis=1)
    else:
        data1 = pd.pivot_table(data, values='predicted_hears_exact',
                               index=['mouse_id', 'mouse_group', 'frequency', 'threshold'],
                               columns=['sound_level'],
                               aggfunc=np.sum).reset_index().rename_axis(None, axis=1)

    # Add model sound levels which are not present in input data
    for i in _sound_levels:
        if i not in data1.columns:
            data1[i] = np.nan

    # Fill NA cells with interpolation:
    data1[_sound_levels] = data1[_sound_levels].interpolate(axis=1, method='linear', limit_direction='both')

    # Name columns
    new_names = [(i, 'sl' + str(i)) for i in data1[_sound_levels].columns.values]
    data1.rename(columns=dict(new_names), inplace=True)

    return data1


def one_hot_encode_frequencies(_data, _poss_frequencies):
    """
    Encodes frequencies as multi-dimensional binary arrays.
    :param _data: pandas-data-frame
        A data frame with frequency values given by the 'frequency'-column.
    :param _poss_frequencies: list
        List of valid frequencies.
    :return: pandas-data-frame
        The input data frame with additional columns for the corresponding frequency binary-encodings.
    """
    # Create dictionaries for frequencies
    freq_dict = dict(zip(_poss_frequencies, list(range(len(_poss_frequencies)))))
    print('frequency dictionary: {}'.format(freq_dict))
    # Hot encode multi-labels
    freq_len = len(_poss_frequencies)
    # Add column with dictionary mappings of sound levels
    _data['freq_index'] = _data['frequency'].map(freq_dict)
    # Add columns with one-hot encodings of sound level
    temp = pd.DataFrame(keras.utils.to_categorical(_data['freq_index'], freq_len))
    temp.columns = ['fr' + str(col) for col in temp.columns]
    _data = pd.concat([_data, temp], axis=1)

    return _data


def one_hot_encode_sound_levels(_data, _poss_thresholds):
    """
    Encodes sound levels as multi-dimensional binary arrays.
    :param _data: pandas-data-frame
        A data frame with sound levels given by the 'sound_level'-column.
    :param _poss_thresholds: list
        List of valid sound levels and thus also valid hearing thresholds.
    :return: pandas-data-frame
        The input data frame with additional columns for the corresponding sound level binary-encodings.
    """
    # Create dictionaries for sound levels
    sl_dict = dict(zip(_poss_thresholds, list(range(len(_poss_thresholds)))))
    print('sound level dictionary: {}'.format(sl_dict))
    # Hot encode multi-labels
    sl_len = len(_poss_thresholds)

    # Add column with dictionary mappings of sound levels
    _data['sound_level_index'] = _data['sound_level'].map(sl_dict)

    # Add columns with one-hot encodings of sound level
    temp = pd.DataFrame(keras.utils.to_categorical(_data['sound_level_index'], sl_len))
    temp.columns = ['sl' + str(col) for col in temp.columns]
    _data = pd.concat([_data, temp], axis=1)

    return _data


def multiple_hot_encode_threshold(_data, _sound_levels, _aul_sound_level):
    """
    First encodes the hearing thresholds by the corresponding position in the list of valid sound levels
    and then converts the numerical labels to binary vectors.
    :param _data: pandas-data-frame
        A data frame with hearing thresholds given by the 'threshold'-column.
    :param _sound_levels: list
        List of valid sound levels and thus also valid hearing thresholds.
    :param _aul_sound_level: int
        The value to be assigned when the manually determined hearing threshold is missing.
    :return: pandas-data-frame
        The input data frame with additional columns for the corresponding threshold binary-encodings.
    """
    # Create new dataframe for further formatting
    data = _data.copy()
    # Check if max sound level is equal to max threshold
    if _aul_sound_level != max(data['threshold']):
        print('Warning: AUL sound level not equal to max threshold!')
    # Create dictionary for sound levels
    temp = [_aul_sound_level] + sorted(_sound_levels, reverse=True)
    thresh_dict = dict(zip(temp, list(range(len(temp)))))
    length_thresholds = len(temp)

    # Add column with dictionary mappings of threshold
    data['thresh_index'] = data['threshold'].map(thresh_dict)

    # Add columns with one-hot encodings of threshold numerical labels
    temp = pd.DataFrame(keras.utils.to_categorical(data['thresh_index'], num_classes=length_thresholds))
    temp.replace(0., np.NaN, inplace=True)
    temp.fillna(method='bfill', axis=1, inplace=True)
    temp.replace(np.NaN, 0, inplace=True)
    temp = temp.apply(pd.to_numeric, downcast='integer')
    temp.columns = ['thr' + str(col) for col in temp.columns]
    data1 = pd.concat([data, temp], axis=1)

    return data1


def get_col_names4model_1_labels(_data):
    """
    Returns the names of the columns containing labels and data with which the first neural network is fed.
    :param _data: pandas-data-frame
        A data frame with binary encodings for frequencies, sound levels and hearing thresholds.
    :return:
        - the name of the main label column
        - the names of the columns with frequency binary encodings
        - the names of the columns with sound level binary encodings
        - the names of the columns with the numerical values describing the ABR hearing curves.
    """
    # Get column names of labels and data
    # Get label column(s) and their lengths
    main_label_col = ["still_hears"]
    freq_label_col = []
    sl_label_col = []
    fr = re.compile('fr\d+')
    freq_label_col += [col for col in _data if fr.match(col)]
    sl = re.compile('sl\d+')
    sl_label_col += [col for col in _data if sl.match(col)]

    # Get data columns (get columns which start with a 't' and numbers after that)
    r = re.compile('t\d+')
    data_cols = [col for col in _data if r.match(col)]
    data_cols = data_cols[:1000]

    return main_label_col, freq_label_col, sl_label_col, data_cols


def get_col_names4model_2_labels(_data):
    """
    Returns the names of the columns containing labels and data used to feed the second neural network.
    :param _data: pandas-data-frame
        A data frame with binary encodings for frequencies, sound levels and hearing thresholds.
    :return:
        - the names of the main label columns
        - the names of the frequency label columns
        - the names of the columns with numerical data that are the sound level hearing predictions of the first neural network.
    """
    # Initialize
    freq_label_col = []

    # Get label column(s) and their lengths
    # Get main columns (get columns which start with a 'thr' and numbers after that)
    thr = re.compile('thr\d+')
    main_label_col = [col for col in _data if thr.match(col)]
    # Get frequency columns (get columns which start with a 'fr' and numbers after that)
    fr = re.compile('fr\d+')
    freq_label_col += [col for col in _data if fr.match(col)]

    # Get data columns (get columns which start with a 'sl' and numbers after that)
    sl = re.compile('sl\d+')
    data_cols = [col for col in _data if sl.match(col)]

    return main_label_col, freq_label_col, data_cols


def standardize(_data, _train_indices, _data_cols):
    """
    Standardizes all input data by mean and standard deviation of the training data.
    :param _data: pandas-data-frame
        A data frame that contains ABR hearing curves plus metadata.
        Each row contains:
        - an identifier of the mouse given by the 'mouse_id'-column
        - the stimulus sound level given by the 'sound_level'-column
        - the stimulus frequency given by the 'frequency'-column
        - the ABR hearing curve recorded for a mouse at given sound_level and frequency.
    :param _train_indices: numpy-array
        The indices of the hearing curves in the input data frame used to train the neural network.
    :param _data_cols: list
        Names of the columns containing the numerical values that describe the hearing curves.
    :return: pandas-data-frame
        A data frame with the standardized input data.
    """
    # Create new dataframe for further formatting
    data = _data.copy()
    # Optional: standardize (z-transformation) all input data by mean and sd of training data
    train_mean = data.iloc[_train_indices,][_data_cols].unstack().mean()
    train_sd = data.iloc[_train_indices,][_data_cols].unstack().std()

    data[_data_cols] = data[_data_cols].add(-train_mean)
    data[_data_cols] = data[_data_cols].div(train_sd)

    print('train mean: {}'.format(train_mean))
    print('train standard deviation: {}'.format(train_sd))

    return data


def split_data(_data, _random_seed):
    """
    Randomly divides the array of mouse IDs into two groups: 80% for training, 20% for testing.
    :param _data: pandas-data-frame
        A data frame that contains ABR hearing curves plus metadata.
        Each row must contain an identifier of the mouse to be examined, given by the 'mouse_id'-column.
    :param _random_seed: int
        The start number of the random number generator.
    :return: numpy-arrays
        - the mouse IDs of the training hearing curves
        - the mouse IDs of the testing hearing curves
        - the indices of the training hearing curves in the input data frame
        - the indices of the testing hearing curves in the input data frame.
    """
    # Randomize the mice
    length_mice = _data.mouse_id.nunique()
    mice = _data.mouse_id.unique()
    np.random.seed(_random_seed)
    np.random.shuffle(mice)

    # Sample the row indices to distinguish between training and test indices
    # Divide the indices
    perc80 = int(math.floor(0.8 * length_mice))

    train_mice = mice[:perc80]
    test_mice = mice[perc80:]

    train_indices = _data.index[_data['mouse_id'].isin(train_mice)]
    test_indices = _data.index[_data['mouse_id'].isin(test_mice)]

    return train_mice, test_mice, train_indices, test_indices


def prepare_data4training_1(_data, _poss_frequencies, _poss_thresholds, _ING_model=False):
    """
    Prepares data to feed the first neural network.
    :param _data: pandas-data-frame
        A data frame that contains ABR hearing curves plus metadata.
        Each row contains:
        - an identifier of the mouse, given by the 'mouse_id'-column
        - the stimulus sound level, given by the 'sound_level'-column
        - the stimulus frequency, given by the 'frequency'-column
        - the ABR hearing curve recorded for a mouse at given sound_level and frequency
        - if existing, the 'validated'-column indicates whether the assigned hearing threshold has been validated by
        several experts or not.
    :param _poss_frequencies: list
        List of valid frequencies.
    :param _poss_thresholds: list
        List of valid sound levels and thus also valid ABR thresholds.
    :param _ING_model: boolean
        True if the neural network is trained on ABR data from Ingham et al., default is false.
    :return:
        - a data frame with preprocessed input data
        - in the case that the network is trained on data from the German Mouse Clinic:
        the name of the column in which the sample weights are stored
    """
    if not _ING_model:
        data, sample_weight_col = weight_sample4model_1(_data)

        data1 = one_hot_encode_frequencies(data, _poss_frequencies)
        data2 = one_hot_encode_sound_levels(data1, _poss_thresholds)

        return data2, sample_weight_col
    else:
        data1 = one_hot_encode_frequencies(_data, _poss_frequencies)
        data2 = one_hot_encode_sound_levels(data1, _poss_thresholds)

        return data2


def prepare_data4training_2(_data, _sound_levels, _aul_sound_level, _ING_model=False):
    """
    Prepares data to feed the second neural network.
    :param _data: pandas-data-frame
        A data frame that contains sound level hearing predictions of the first neural network plus metadata
        such as mouse ID, stimulus frequency, stimulus sound level and manually determined hearing threshold.
    :param _sound_levels: list
        List of valid sound levels.
    :param _aul_sound_level: int
        The value to be assigned when the manually determined hearing threshold is missing.
    :param _ING_model: boolean
        True if the neural network is trained on ABR data from Ingham et al., default is false.
    :return:
        - a data frame with preprocessed input data
        - in the case that the network is trained on data from the German Mouse Clinic:
        the name of the column in which the sample weights are stored
    """
    if not _ING_model:
        data, sample_weight_col = weight_sample4model_2(_data)
        data1 = preprocess_data4model_2(data, _sound_levels, _aul_sound_level)
    else:
        data1 = preprocess_data4model_2(_data, _sound_levels, _aul_sound_level)

    data2 = multiple_hot_encode_threshold(data1, _sound_levels, _aul_sound_level)
    data3 = one_hot_encode_frequencies(data2, sorted(data2['frequency'].unique()))

    if _ING_model:
        return data3
    else:
        return data3, sample_weight_col
