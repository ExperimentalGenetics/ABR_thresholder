"""Given a dataset of ABR hearing curves, this script enables the detection of ABR hearing thresholds for a given
stimulus frequency."""

import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

# Define possible frequency values
poss_freq = [100, 6000, 12000, 18000, 24000, 30000]

# Define possible threshold values
poss_thr = [p for p in range(0, 100, 5)]

# Mean and standard deviation of training data from the German Mouse Clinic
train_mean = -0.011975352770601522
train_sd = 0.36899869016381637

# Mean and standard deviation of training data from Ingham et al.
train_mean_ING = -0.0010400656057877457
train_sd_ING = 0.49735198404515524

# Define cutoff for threshold
threshold_cutoff = 0.5

# Define the value to be assigned if hearing thresholds cannot be determined
aul_sound_level = 999


def make_curve_specific_predictions(_data, _model, _poss_thr=poss_thr, _ING_model=False):
    """
    Makes predictions about whether a given mouse will hear a given stimulus.
    :param _ING_model: boolean
        True if the model was trained with ABR data from Ingham et al..
    :param _poss_thr: list
        List of valid thresholds.
    :param _data: pandas-data-frame
        A data frame that contains an ABR wave plus metadata in each row.
        Each row contains:
        - an identifier of the mouse given by the 'mouse_id'-column
        - the stimulus sound level given by the 'sound_level'-column
        - the ABR wave that was recorded for the mouse-stimulus pair.
        The columns of the series are given as a list in time_series.
    :param _model: Keras model
        The first neural network of the ABR-thresholder that was trained.
    :return: pandas-data-frame
        A data frame with columns for the mouse ID, the stimulus frequency, the stimulus sound level and the hearing prediction.
    """
    # Check if the given thresholds make sense
    if 'threshold' in _data.columns:
        data1 = _data.loc[(_data['threshold'].isin(_poss_thr + [aul_sound_level]))]
    else:
        data1 = _data.copy()

    # Check if column names are present/have the correct name - for meta info
    if not {'mouse_id', 'frequency', 'sound_level'}.issubset(data1.columns):
        raise ValueError('Input data does not contain all necessary columns (or has wrong names)!')

    # Check if column names are present/have the correct name - for data info
    if not {'t' + str(i) for i in range(0, 1000)}.issubset(data1.columns):
        raise ValueError('Input data does not contain all necessary columns (or has wrong names)!')

    # Define curve meta information and the first 1000 value columns
    meta_cols = ['mouse_id', 'frequency', 'sound_level']
    if 'threshold' in data1.columns:
        meta_cols = meta_cols + ['threshold']
    if 'mouse_group' in data1.columns:
        meta_cols = meta_cols + ['mouse_group']

    data_cols = ['t' + str(i) for i in range(0, 1000)]

    # Standardize overall
    if _ING_model:
        _train_mean = train_mean_ING
        _train_sd = train_sd_ING
    else:
        _train_mean = train_mean
        _train_sd = train_sd
        
    print('standardize overall\n train_mean = %s\n train_sd = %s' % (_train_mean, _train_sd))
    data1[data_cols] = data1[data_cols].add(-_train_mean)
    data1[data_cols] = data1[data_cols].div(_train_sd)

    # Curve-specific predictions
    predictions_tmp = _model.predict(data1[data_cols].values[:, :, np.newaxis])
    predictions = pd.DataFrame(predictions_tmp[0])
    predictions.columns = ['predicted_hears_exact']
    data2 = pd.concat([data1[meta_cols], predictions], axis=1)

    return data2[meta_cols + ['predicted_hears_exact']]


def make_threshold_predictions(_data, _model, _poss_thr=poss_thr):
    """
    Predicts the ABR hearing thresholds from the curve specific predictions.
    :param _poss_thr: list
        List of valid thresholds
    :param _data: pandas-data-frame
        A data frame containing the curve specific predictions of the first neural network.
    :param _model: Keras model
        The second neural network of the ABR-thresholder that was trained.
    :return: pandas-data-frame
        A data frame with the predicted threshold per mouse and stimulus.
    """

    print(_poss_thr)

    # Copy output of first model
    data = _data.copy()

    # Raise error if sound levels of input data are not all in model sound levels
    if any(elem not in _poss_thr for elem in list(data['sound_level'].unique())):
        raise ValueError('Input data has more sound levels than the model')

    index = ['mouse_id', 'frequency']
    if 'threshold' in data.columns:
        index = index + ['threshold']
    if 'mouse_group' in data.columns:
        index = index + ['mouse_group']

    # Long to wide format for sound level column
    data1 = pd.pivot_table(data, values='predicted_hears_exact',
                           index=index,
                           columns=['sound_level'],
                           aggfunc=np.sum).reset_index().rename_axis(None, axis=1)

    # Add model sound levels which are not present in input data
    for i in _poss_thr:
        if i not in data1.columns:
            data1[i] = np.nan

    # Fill NA cells
    # With interpolation:
    data1[_poss_thr] = data1[_poss_thr].interpolate(axis=1, method='linear', limit_direction='both')

    # Name columns
    # data2.rename(columns={'Sound Level (dB)': ''})
    new_names = [(i, 'sl' + str(i)) for i in data1[_poss_thr].columns.values]
    data1.rename(columns=dict(new_names), inplace=True)

    # Get data columns for the second model(get columns which start with a 'sl' and numbers after that)
    sl = re.compile('sl\d+')
    data_cols = [col for col in data1 if sl.match(col)]

    # Prediction
    # Predict
    predictions_tmp = _model.predict(data1[data_cols].values[:, :, np.newaxis])
    predictions = pd.DataFrame(predictions_tmp[0])

    # Define column names of predictions
    predictions.columns = ['thr' + str(col) for col in predictions.columns]

    # Make the threshold cutoff from the vector
    a = predictions.where(predictions > threshold_cutoff).notna()
    a = a[a == True]
    b = a.apply(lambda x: x[x.notnull()].index.values[-1], axis=1)

    # Save estimated probabilities
    data1['nn_predict_probs'] = [predictions.loc[idx, b.loc[idx]] for idx in b.index]

    c = b.str.replace(r'\D+', '').astype('int')

    thr_dict = dict(zip(list(range(len(_poss_thr[::-1]))), _poss_thr[::-1]))

    # Decode the threshold encoding
    data1['predicted_thr'] = c.map(thr_dict)

    pred_columns = ['predicted_thr', 'nn_predict_probs']
    if 'threshold' in data.columns:
        for idx in data1.index:
            thr = data1.at[idx, 'threshold']
            if thr == aul_sound_level:
                data1.at[idx, 'nn_thr_score'] = 0.0
            else:
                data1.at[idx, 'nn_thr_score'] = predictions.at[idx, 'thr' + str(list(thr_dict.values()).index(thr))]
        pred_columns = pred_columns + ['nn_thr_score']

    # Define column order
    columns = index + pred_columns + data_cols

    data1 = data1[columns]

    return data1


def print_overall_mouse_metrics(_data,
                                _manual_thr_col='threshold',
                                _predicted_thr_col='predicted_thr'):
    """
    Prints different metrics of the predictions.
    :param _data: pandas-data-frame
        A data frame containing both the manually determined thresholds and the thresholds detected with neural networks.
    :param _manual_thr_col: string
        The name of the column with manually determined thresholds.
    :param _predicted_thr_col: string
        The name of the column in which thresholds detected with neural networks are stored.
    :return: None
    """
    result_test = _data.copy()
    result_test = result_test.sort_values(by='frequency')

    # Compute distance to actual threshold
    result_test['thr_dist'] = abs(result_test[_manual_thr_col] - result_test[_predicted_thr_col])

    # Print exact accuracy of main label prediction
    thr_acc = 100 * round(len(result_test[result_test['thr_dist'] == 0]) / len(result_test), 3)
    print('Threshold accuracy: %.2f%%' % thr_acc)
    freq_thr_acc = []
    for freq in result_test.frequency.unique():
        result_test1 = result_test[result_test.frequency == freq]
        thr_acc1 = 100 * round(len(result_test1[result_test1['thr_dist'] == 0]) / len(result_test1), 3)
        freq_thr_acc.append(thr_acc1)
        if freq == 100:
            print("   Click: %.2f%%" % thr_acc1)
        else:
            print("   Frequency %d: %.2f%%" % (freq, thr_acc1))
    print('Averaged threshold accuracy: %.2f%%\n' % np.average(np.array(freq_thr_acc)))

    # Print error of main label prediction
    all_mae = round(np.mean(result_test['thr_dist']), 3)
    print('Mean error of all classifications: %.2f\n' % all_mae)

    # Print error only of wrong classifications
    result_test2 = result_test[result_test[_manual_thr_col] != result_test[_predicted_thr_col]]
    f_mae = round(
        np.mean(abs(result_test2[_manual_thr_col] - result_test2[_predicted_thr_col])), 3)
    print("Absolute mean error of false classifications: %.2f" % f_mae)
    for freq in result_test.frequency.unique():
        result_test3 = result_test2[result_test2.frequency == freq]
        freq_f_mae = round(np.mean(abs(result_test3[_manual_thr_col] - result_test3[_predicted_thr_col])), 3)
        freq_mae = round(np.mean(abs(result_test[result_test.frequency == freq][_manual_thr_col] -
                                     result_test[result_test.frequency == freq][_predicted_thr_col])), 3)
        if freq == 100:
            print("   Click: %.2f (%.2f)" % (freq_f_mae, freq_mae))
        else:
            print('   Frequency %d: %.2f (%.2f)' % (freq, freq_f_mae, freq_mae))

    # Print accuracy with 5 db buffer
    thr_acc_5db = 100 * round(len(result_test[result_test.thr_dist <= 5]) / len(result_test)
                              , 3)
    print('\nThreshold accuracy with 5dB tolerance: %.2f%%' % thr_acc_5db)
    freq_thr_acc_5db = []
    mice_count = []
    for freq in result_test.frequency.unique():
        result_test1 = result_test[result_test.frequency == freq]
        thr_acc1 = 100 * round(len(result_test1[result_test1.thr_dist <= 5]) /
                               len(result_test1), 3)
        freq_thr_acc_5db.append(thr_acc1)
        count = result_test1.mouse_id.nunique()
        mice_count.append(count)
        if freq == 100:
            print('   Click: %.2f%% (%d)' % (thr_acc1, count))
        else:
            print('   Frequency %d: %.2f%% (%d)' % (freq, thr_acc1, count))
    print('Averaged threshold accuracy with 5dB tolerance: %.2f%%\n' % np.average(
        np.array(freq_thr_acc_5db), weights=np.array(mice_count)))

    # Print accuracy with 10 db buffer
    thr_acc_10db = 100 * round(len(result_test[result_test.thr_dist <= 10]) / len(result_test)
                               , 3)
    print('\nThreshold accuracy with 10dB tolerance: %.2f%%' % thr_acc_10db)
    freq_thr_acc_10db = []
    mice_count = []
    for freq in result_test.frequency.unique():
        result_test1 = result_test[result_test.frequency == freq]
        thr_acc1 = 100 * round(len(result_test1[result_test1.thr_dist <= 10]) /
                               len(result_test1), 3)
        freq_thr_acc_10db.append(thr_acc1)
        count = result_test1.mouse_id.nunique()
        mice_count.append(count)
        if freq == 100:
            print('   Click: %.2f%% (%d)' % (thr_acc1, count))
        else:
            print('   Frequency %d: %.2f%% (%d)' % (freq, thr_acc1, count))
    print('Averaged threshold accuracy with 10dB tolerance: %.2f%%\n' % np.average(
        np.array(freq_thr_acc_10db), weights=np.array(mice_count)))


def plot_curves_per_mouse(_data, _mouse_id, _data_cols, _file=None, _data_range=range(0, 1000)):
    """
    Plots the ABR hearing curves recorded for a given mouse ID. The plots may be saved to a file.
    :param _data: pandas-data-frame
        A data frame that contains an ABR wave plus metadata in each row.
    :param _mouse_id: int or string
        The mouse identifier for which hearing curves are to be plotted.
    :param _data_cols: list
        The list of columns in which the time series are stored.
    :param _file: string
        The name of the file in which the plots are saved, default is None.
    :param _data_range: numpy-array
        The time steps.
    :return: None
    """
    fig = plt.figure(constrained_layout=True, figsize=(80, 64))

    sound_levels = _data['sound_level'].unique()
    df = _data[_data.mouse_id == _mouse_id]

    cols = 2
    rows = int(len(df.frequency.unique()) / cols)
    col = 0
    row = 0
    spec = gridspec.GridSpec(ncols=cols, nrows=rows, figure=fig)
    f_ax = {}

    for idx, freq in enumerate(df.frequency.unique()):

        f_ax[idx] = fig.add_subplot(spec[row, col])
        f_ax[idx].set_title('%sHz' % freq)
        f_ax[idx].set_yticks(sound_levels)

        # Compute thresholds for horizontal line
        nn_thr = None
        slr_thr = None
        human_thr = None
        if 'nn_predicted_thr' in df.columns:
            thr = df[df['frequency'] == freq]['nn_predicted_thr'].unique()
            if len(thr) > 0:
                nn_thr = thr[0]
        if 'slr_estimated_thr' in df.columns:
            thr = df[df['frequency'] == freq]['slr_estimated_thr'].unique()
            if len(thr) > 0:
                slr_thr = thr[0]
        thr = df[df['frequency'] == freq]['threshold'].unique()
        if len(thr) > 0:
            human_thr = thr[0]
        # Plot
        plt.rcParams.update({'font.size': 20})
        f_ax[idx].set_xlabel('Timesteps (overall 10ms)')
        f_ax[idx].set_ylabel('Corresponding sound level in dB')
        f_ax[idx].set_title('%dHz - %sdB (human), %sdB (NN: neural network), %sdB (SLR - sound level regression)' % (
            freq, human_thr, nn_thr, slr_thr))

        for sound_level in df.loc[df['frequency'] == freq, 'sound_level']:
            f_ax[idx].plot(_data_range, sound_level +
                           2.5 * df[(df['sound_level'] == sound_level) & (df['frequency'] == freq)][_data_cols].iloc[0],
                           linewidth=2.5)

        if human_thr and human_thr != 999:
            f_ax[idx].hlines(y=human_thr,
                             xmin=_data_range[0], xmax=_data_range[-1],
                             linewidth=2.5, linestyles='dotted')
        if nn_thr and nn_thr != 999:
            f_ax[idx].hlines(y=nn_thr,
                             xmin=_data_range[0], xmax=_data_range[-1],
                             linewidth=2.5, linestyles='dotted')
        if slr_thr and slr_thr != 999:
            f_ax[idx].hlines(y=slr_thr,
                             xmin=_data_range[0], xmax=_data_range[-1],
                             linewidth=2.5, linestyles='dotted')

        col += 1
        if col == cols:
            row += 1
            col = 0

        labels = ['thr NN/SLR/human = %d' % sl if (sl == human_thr and sl == nn_thr and sl == slr_thr)
                  else 'thr NN/SLR = %d' % sl if (sl == nn_thr and sl == slr_thr)
                  else 'thr NN/human = %d' % sl if (sl == human_thr and sl == nn_thr)
                  else 'thr SLR/human = %d' % sl if (sl == human_thr and sl == slr_thr)
                  else 'thr NN = %d' % sl if sl == nn_thr
                  else 'thr SLR = %d' % sl if sl == slr_thr
                  else 'thr human = %d' % sl if sl == human_thr
                  else sl for sl in sound_levels]
        f_ax[idx].yaxis.set_major_formatter(ticker.FixedFormatter(labels))

    fig.suptitle('Mouse ID: %s' % _mouse_id, fontsize=24)
    if _file:
        plt.savefig(_file)
