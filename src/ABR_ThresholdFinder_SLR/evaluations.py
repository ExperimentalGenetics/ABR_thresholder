import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _prepare_thresholds_for_evaluation(data, db_buffer, sound_level='sound_level', threshold='threshold',
                                       threshold_cleaned='threshold_cleaned'):
    """
    Changes threshold.max() -> max sound level + db_buffer and min threshold.min -> sound_level.min()

    Parameters
    ----------
        data : pandas-data-frame
            A data frame that contains an ABR wave metadata in each row. 
           
        mouse_id : string, default 'mouse_id'
            Name of column with mouse_ids in data.
            
        sound_level : string, default 'sound_level'
            Name of column with sound_levels in data.
            
        threshold : string, default 'threshold'
            Name of column with threshold in data.
            
        threshold_cleaned : string, default 'threshold_cleaned'
            Name of column in which the cleaned thresholds should be placed.
    Returns
    -------
    data : pandas-data-frame
        The input data frame with an additional column threshold_cleaned that contains the prepared threshold.
    """

    # print('data:')
    # print(data[['mouse_id', sound_level, threshold]])

    data[threshold_cleaned] = data[threshold]

    all_mice = data['mouse_id'].unique()

    for mouse in all_mice:
        maxsl = data.loc[data['mouse_id'] == mouse, sound_level].max()
        minsl = data.loc[data['mouse_id'] == mouse, sound_level].min()

        data.loc[data['mouse_id'] == mouse, threshold_cleaned] = min(maxsl + db_buffer, data.loc[
            data['mouse_id'] == mouse, threshold_cleaned].max())
        data.loc[data['mouse_id'] == mouse, threshold_cleaned] = max(minsl, data.loc[
            data['mouse_id'] == mouse, threshold_cleaned].max())

    return data


def _evaluate_classification_against_ground_truth_for_specific_frequency(
        data, freq, db_buffer,
        frequency='frequency',
        mouse_id='mouse_id',
        sound_level='sound_level',
        threshold_estimated='threshold_estimated',
        threshold_ground_truth='threshold'):
    """
    Compares a threshold estimation against a ground truth for the thresholds for a single frequency stimulus.
    To be able to compute the 'Root Mean squared threshold error' and the 
    'Percentage of correctly predicted mice, when allowing for a db_buffer error'
    The maximum threshold is mapped to the maximum sound level + db_buffer
    and the minimum threshold is mapped to the minimum sound level - db_buffer

    Parameters
    ----------
        data : pandas-data-frame
            A data frame that contains an ABR wave metadata in each row. 
            
        freq : int
            Single frequency stimulus for which the evalution should be done.

        db_buffer: int
            Allowed margin of error for corrected predicted thresholds.
            
        frequency : string, default 'mouse_id'
            Name of column with frequency in data.

        mouse_id : string, default 'mouse_id'
            Name of column with mouse_ids in data.
            
        sound_level : string, default 'sound_level'
            Name of column with sound_levels in db in data.
            
        threshold : string, default 'threshold'
            Name of column with threshold in data.
            
        threshold_cleaned : string, default 'threshold_cleaned'
            Name of column in which the cleaned thresholds should be placed.

    Returns
    -------
    evaluation : dictionary
        Dictionary with 
            +Percentage of correctly predicted mice
            +Percentage of correctly predicted mice, when allowing for a db_buffer error
            +Root Mean squared threshold error
            
    """

    # filter data for certain frequency
    selection = (data.frequency == freq)
    data_freq = data[selection]

    data_freq = _prepare_thresholds_for_evaluation(data_freq, db_buffer,
                                                   sound_level=sound_level,
                                                   threshold=threshold_estimated,
                                                   threshold_cleaned=threshold_estimated + '_cleaned')

    data_freq = _prepare_thresholds_for_evaluation(data_freq, db_buffer,
                                                   sound_level=sound_level,
                                                   threshold=threshold_ground_truth,
                                                   threshold_cleaned=threshold_ground_truth + '_cleaned')

    data_freq_groupedbymouse = data_freq.groupby(mouse_id).min()

    # print()
    # print(data_freq[threshold_estimated + '_cleaned'] - data_freq[threshold_ground_truth + '_cleaned'])

    threshold_error = np.sqrt(
        (data_freq[threshold_estimated + '_cleaned'] - data_freq[
            threshold_ground_truth + '_cleaned']).to_numpy().astype(float) ** 2)
    # threshold_error = np.sqrt(
    #    (data_freq[threshold_estimated] - data_freq[threshold_ground_truth]).to_numpy().astype(float) ** 2)

    percentage_of_correctly_predicted_mice = np.mean(threshold_error == 0) * 100
    percentage_of_correctly_predicted_mice_with_db_buffer = np.mean(threshold_error <= db_buffer) * 100
    root_mean_squared_threshold_error = threshold_error.mean()

    print('Percentage of correctly predicted mice %d%%' % percentage_of_correctly_predicted_mice)
    # print(
    #     'Percentage of correctly predicted mice, when allowing for a 5db error %d%%' % percentage_of_correctly_predicted_mice_with_db_buffer)
    print('Percentage of correctly predicted mice, when allowing for a {}db error {}'.format(db_buffer,
                                                                                             percentage_of_correctly_predicted_mice_with_db_buffer))
    print('Root Mean squared threshold error %f' % root_mean_squared_threshold_error)

    return {'percentage_of_correctly_predicted_mice': percentage_of_correctly_predicted_mice,
            'percentage_of_correctly_predicted_mice_with_' + str(
                db_buffer) + 'db_buffer': percentage_of_correctly_predicted_mice_with_db_buffer,
            'root_mean_squared_threshold_error': root_mean_squared_threshold_error}


def evaluate_classification_against_ground_truth(data, db_buffer,
                                                 frequency='frequency',
                                                 mouse_id='mouse_id',
                                                 sound_level='sound_level',
                                                 threshold_estimated='threshold_estimated',
                                                 threshold_ground_truth='threshold'):
    """
    Compares a threshold estimation against a ground truth for the thresholds.
    To be able to compute the 'Root Mean squared threshold error' and the 
    'Percentage of correctly predicted mice, when allowing for a 5db error'
    The maximum threshold is mapped to the maximum sound level + 5db
    and the minimum threshold is mapped to the minimum sound level - 5db

    Parameters
    ----------
        data : pandas-data-frame
            A data frame that contains an ABR wave metadata in each row. 
            
        frequency : string, default 'mouse_id'
            Name of column with frequency in data.

        mouse_id : string, default 'mouse_id'
            Name of column with mouse_ids in data.
            
        sound_level : string, default 'sound_level'
            Name of column with sound_levels in db in data.
            
        threshold : string, default 'threshold'
            Name of column with threshold in data.
            
        threshold_cleaned : string, default 'threshold_cleaned'
            Name of column in which the cleaned thresholds should be placed.
    Returns
    -------
    evaluation : pandas-data-frame
        Pandas dataframe with evaluations for each frequency.         
        Columns are:
            +Percentage of correctly predicted mice
            +Percentage of correctly predicted mice, when allowing for a 5db error
            +Root Mean squared threshold error
            
    """

    evaluation = pd.DataFrame(columns=['Frequency', 'percentage_of_correctly_predicted_mice',
                                       'percentage_of_correctly_predicted_mice_with_' + str(db_buffer) + 'db_buffer'])

    for freq in data.frequency.unique():
        print(f'Stimulus: {freq}')
        evaluation_freq = _evaluate_classification_against_ground_truth_for_specific_frequency(
            data, freq, db_buffer,
            frequency=frequency,
            mouse_id=mouse_id,
            sound_level=sound_level,
            threshold_estimated=threshold_estimated,
            threshold_ground_truth=threshold_ground_truth);
        evaluation_freq['Frequency'] = freq

        evaluation = evaluation.append(evaluation_freq, ignore_index=True)

    return evaluation


def _generate_evaluation_curve(data, thresholds, time_series, sound_level='sound_level'):
    """
    Computes and evaluation curves for a specific stimulus.
    
    For each ABR wave a threshold_normalized_sound_level = sound_level/threshold
    is computed.
    Then the ABR waves are sorted with respect to the threshold_normalized_sound_level in increasing order.
    
    Then we compute the strength of signal average over the N ABR-curves with smallest threshold_normalized_sound_level
    and plot it against N.
    
    If the given thresholds are correct, there should be no signal strength for all curves with
    threshold_normalized_sound_level < 1.
    
    To compute the evaluation curves the maximum threshold is mapped to the maximum sound level + 5db
    and the minimum threshold is mapped to the minimum sound level - 5db.
    

    Parameters
    ----------
        data : pandas-data-frame
            A data frame that contains an ABR wave plus metadata in each row. 
            
        thresholds : string or int
            Name of column with thresholds in data.
            Column names where the thresholds are stored for which the evaluation
            should be computed. If int, then a constant threshold with this value is used.
            
        time_series : list of strings
            A list of strings, where each string is a columns name in data that contains the ABR value at a certain point in time.
            
        sound_level : string, default 'sound_level'
            Name of column with sound_levels in data.
            
    Returns
    -------
        subthsignal_strength : numpy array shape (Number of ABR waves)
            Strength of signal average over the N ABR-curves with smallest threshold_normalized_sound_level.
        N : numpy array shape (Number of ABR waves)
            The corresponding number of ABR-curves with smallest threshold_normalized_sound_level.
    """

    if type(thresholds) is str:
        data = _prepare_thresholds_for_evaluation(data, 5,
                                                  sound_level=sound_level,
                                                  threshold=thresholds,
                                                  threshold_cleaned=thresholds + '_cleaned')
        thresholds = data[thresholds + '_cleaned']

    # print(thresholds)

    subthsignal_strength = []
    subthsignal_strength_std = []
    count = []
    begin = True
    th_old = -np.inf
    current_count = 0
    s = 0
    for tha in np.arange(-3, 3, 0.1):
        th = thresholds * np.exp(tha)
        current_count += ((data['sound_level'] >= th_old) & (data['sound_level'] < th)).sum()

        s += data.loc[(data['sound_level'] >= th_old) & (data['sound_level'] < th), time_series].sum()

        th_old = th

        s_mean = s / current_count

        signal_sub = s_mean.var()

        count.append(current_count)
        subthsignal_strength.append(signal_sub)

    subthsignal_strength = np.array(subthsignal_strength)
    N = np.array(count)
    return subthsignal_strength, N


def plot_evaluation_curve_for_specific_stimulus(data, freq, thresholds, time_series,
                                                frequency='frequency',
                                                sound_level='sound_level',
                                                fontsize=20,
                                                xlabel='Normalized N',
                                                ylabel='Normalized strength of signal average\nover the N ABR-curves\nwith smallest threshold_normalized_sound_level',
                                                legend=True,
                                                ax=None):
    """
    Computes and plots evaluation curves for a specific stimulus.
    
    For each ABR wave a threshold_normalized_sound_level = sound_level/threshold
    is computed.
    Then the ABR waves are sorted with respect to the threshold_normalized_sound_level in increasing order.
    
    Then we compute the strength of signal average over the N ABR-curves with smallest threshold_normalized_sound_level
    and plot it against N.
    
    N and strength of signal are both normalized to have a maximum of 1.
    
    If the given thresholds are correct, there should be no signal strength for all curves with
    threshold_normalized_sound_level < 1.
    
    To compute the evaluation curves the maximum threshold is mapped to the maximum sound level + 5db
    and the minimum threshold is mapped to the minimum sound level - 5db.
    

    Parameters
    ----------
        data : pandas-data-frame
            A data frame that contains an ABR wave plus metadata in each row. 
            
        freq : int
            Single frequency stimulus for which the evalution should be done.
            
        thresholds : list with strings or ints
            List containing strings with columns names where the thresholds are stored for which the evaluations
            should be computed. If int, then a constant threshold with this value is used.
            
        time_series : list of strings
            A list of strings, where each string is a columns name in data that contains the ABR value at a certain point in time.
           
        mouse_id : string, default 'mouse_id'
            Name of column with mouse_ids in data.
            
        sound_level : string, default 'sound_level'
            Name of column with sound_levels in data.
            
    Returns
    -------
    fig : figure
        A handel to the plotted figure.
    """

    if ax is None:
        fig = plt.figure(1, figsize=(15, 15))

    line_styles = ['dotted', 'dashed', 'dashdot']
    thr_line_styles = {}
    thr_colors = {}
    legend_elements = []
    idx = 0
    for threshold in thresholds:
        if type(threshold) is not str:
            thr_line_styles[threshold] = 'solid'
            thr_colors[threshold] = 'dimgray'
        else:
            thr_line_styles[threshold] = line_styles[idx]
            if 'NN' in threshold:
                thr_colors[threshold] = '#4daf4a' #'green'
            elif 'SLR' in threshold:
                thr_colors[threshold] = '#e41a1c' #'red'
            else:
                thr_colors[threshold] = '#377eb8' #'mediumblue'

            idx += 1       
            
    evaluation_curve = {}
    min_number_datapoints = 150
    for threshold in thresholds:

        subthsignal_strength, N = _generate_evaluation_curve(data[data[frequency] == freq], threshold, time_series,
                                                             sound_level='sound_level')

        subthsignal_strength_normalized = subthsignal_strength[N > min_number_datapoints] / subthsignal_strength[
            N > min_number_datapoints].max()
        N_normalized = N[N > min_number_datapoints] / N.max()

        if type(threshold) is not str:
            threshold_label = 'dummy threshold (%s)' % str(threshold)
        else: 
            threshold_label = '%s' % ('manual threshold' if 'manual' in threshold 
                else 'NN predicted threshold' if 'NN' in threshold
                else 'SLR estimated threshold' if 'SLR' in threshold else threshold)
        
        if ax is None:
            plt.plot(N_normalized, subthsignal_strength_normalized, label=threshold_label)
        else:
            ax.plot(N_normalized, subthsignal_strength_normalized, label=threshold_label,
                    color=thr_colors[threshold],
                    linestyle=thr_line_styles[threshold], lw=2)
            
        legend_elements.append(Line2D([0], [0], color=thr_colors[threshold],
                                      linestyle=thr_line_styles[threshold], lw=2, 
                                      label=threshold_label)) 

    if legend:    
        plt.legend(fontsize=fontsize)
    
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=fontsize)
    else: 
        plt.setp(ax.get_xticklabels(), visible=False)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=fontsize)
    else: 
        plt.setp(ax.get_yticklabels(), visible=False)
    
    if ax is None:
        return fig
    else:
        return legend_elements
