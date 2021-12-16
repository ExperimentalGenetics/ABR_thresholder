import pandas as pd
import numpy as np

import progressbar

import statsmodels.api as sm

import multiprocessing

from collections import defaultdict

from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, GroupKFold

import pickle

import warnings

warnings.filterwarnings('ignore')


def _fit_piecewise_function(x, y, th, k_fold, deg):
    """
    Fits a Piecewise function, given a break point.
    Fits a Piecewise function f: x->y constisting of a constant part f(x<th): x -> constant
    and a nonlinear part for x>th.
    For the nonlinear part a polynomial of degree deg is used.
    Fitted happens using Elastic net with k_fold CV.

    Parameters
    ----------
    x : numpy-array with shape (n_samples, )
        Training data.

    y : numpy-array with shape (n_samples, )
        Target_values

    th: float
        Break point of the piecewise function.

    k_fold : int
        k_fold for cross validation in elastic net.

    deg : int
        Degree of polynomial basis for fitting the nonlinear part.

    Returns
    -------
    cv_error: float, shape (total number of ABR waves,)
        Error of the fit as estimated by cross validation.

    y_estim : numpy-array with shape (n_samples, )
        The fitted target values.
    """

    # normalize x and y
    x_norm = (x - th) / x.max()
    y_norm = (y - y.mean()) / y.std()

    # built polynomial basis
    X = np.polynomial.polynomial.polyvander(x_norm, deg)
    X[x_norm < 0, 1:] = 0

    # fit piecewise function
    regr = ElasticNetCV(cv=k_fold, random_state=0, l1_ratio=[0.5, 0.99], fit_intercept=False, n_alphas=10, n_jobs=10)
    regr.fit(X, y_norm)

    # extract cross validation error of best solution
    # print(regr.mse_path_)

    cv_error = regr.mse_path_.mean(axis=-1).min()

    # renormalize fitted values
    y_estim = regr.predict(X) * y.std() + y.mean()
    return cv_error, y_estim


def _loop_through_breakpoints_and_fit_piecewiese_functions(x, y, deg=5, k_fold=5):
    """
    Fits a piecewise function.
    Loops through different potential breakpoint values, computes cross validation error
    for each breakpoints and chooses the breakpoint with the lowest error.

    For each breakpoints value
    fits a piecewise function f: x->y consisting of a constant part f(x<th): x -> constant
    and a nonlinear part for x>th.
    For the nonlinear part a polynomial of degree deg is used.
    Fitted happens using Elastic net with k_fold cross validation.

    Parameters
    ----------
    x : numpy-array with shape (n_samples, )
        Training data.

    y : numpy-array with shape (n_samples, )
        Target_values

    k_fold : int
        k_fold for cross validation in elastic net.

    deg : int
        Degree of polynomial basis for fitting the nonlinear part.

    Returns
    -------
    ths_best : float
        Break point value of best fit.

    y_estim_best : numpy-array with shape (n_samples, )
        Best fit.
        """

    # compute rough estimates of an upper and a lower bound of the threshold to limit the search space
    # upper bound for threshold: largest value x for which y(x<upper bound) is significantly an increasing function
    # (using a conservative significance test the Bonferroni correction)
    max_th = _giv_max_threshold_due_to_significance_test(x, y)
    # lower bound for threshold: use first increase in an isotonic regression
    min_th = _giv_min_threshold_due_to_isotonic_regression(x, y)

    max_th = min(x.max() + 5, max_th)  # make sure max_th is not larger than the largest x-value (x = true sound level)
    min_th = max(0, min(min_th, max_th - 5))  # make sure min_th is strictly smaller than max_th but not smaller than 0

    ths = np.arange(min_th, max_th, 5)
    cv_errors = np.zeros_like(ths)
    maxlikes = []
    N = x.shape[0]

    for i, th in enumerate(ths):
        cv_error, y_estim = _fit_piecewise_function(x, y, th, k_fold, deg)

        maxlikes.append(y_estim)
        cv_errors[i] = cv_error

        # now take a look to the right and left of the best value and see if the optimal threshold is smaller or
        # larger actually
    i_best = cv_errors.argmin()
    th_best = ths[i_best]
    for corr in [-0.5, 0.5]:
        th_corr = th_best + corr
        cv_error, y_estim = _fit_piecewise_function(x, y, th_corr, k_fold, deg)

        if cv_error < cv_errors[i]:
            ths[i_best] = th_corr
            cv_errors[i_best] = cv_error
            maxlikes[i_best] = y_estim

    y_estim_best = maxlikes[i_best]
    ths_best = ths[i_best]

    return ths_best, y_estim_best


def _giv_max_threshold_due_to_significance_test(y, y_guess):
    """
    Gives the smallest leftbound interval in y that has a significant correlation between y and y_guess and a
    significance level of 5%.
    Uses the conservative Bonferroni correction to take into account that we test several intervals for
    statistical significance of positive correlation of y and y_guess.

    Parameters
    ----------
    y : numpy-array with shape (n_samples, )
        Target values

    y_guess : numpy-array with shape (n_samples, )
        Predictor for y.

    Returns
    -------
    max_th : float
        For y< max_th y and y_guess are statistically significantly correlated.
    """

    sort_indices = np.argsort(y)

    y_sorted = y[sort_indices]
    y_guess_sorted = y_guess[sort_indices]

    if len(y_guess_sorted) <= 3:
        return max(y_sorted)

    max_number_of_intervals = len(y_guess_sorted) - 3

    # print(max_number_of_intervals)

    for i in range(max_number_of_intervals):

        exog = sm.add_constant(y_sorted)[:i + 4]
        endog = y_guess_sorted[:i + 4]

        # Fit and summarize OLS model

        mod = sm.OLS(endog, exog)

        res = mod.fit()
        # stopping criterion using Bonferroni, when there is a statstically significant positive correlation
        if (res.pvalues[1] < 0.05 / max_number_of_intervals) & (res.params[1] > 0):
            break

    max_th = exog.max()

    return max_th


def _giv_min_threshold_due_to_isotonic_regression(y, y_guess):
    """
    Gives first jump in istotonic regression of y_guess against y.

    Parameters
    ----------
    y : numpy-array with shape (n_samples, )
        Target values

    y_guess : numpy-array with shape (n_samples, )
        Predictor for y.

    Returns
    -------
    min_th : float
        After y=min_th an isotonic regression of y_guess against y starts increasing.
    """

    # get a minimum threshold using isotonic regression
    ir = IsotonicRegression()
    y_ = ir.fit_transform(y, y_guess)
    min_th = y[
        y_ == y_.min()].max()  # take value after first jump of the isotonic regression as a minimum value for the threshold
    return min_th


class ABR_Threshold_Detector_multi_stimulus:
    """
    ABR Threshold Detector for a multiple stimuli.

    The ABR Threshold Detector uses the "Sound Level Regression" method to detect
    the hearing Thresholds in an ABR data set.
    This class can be used for ABR data that results from multiple stimuli.


    The Sound Level Regression method works in two steps:
    First a regression model (in this case a Random Forest) is trained to predict the exact sound
    level of an ABR curve from its wave form (data is preprocessed by converting to power spectrum and cut of at high
    frequencies)
    . Then a prediction is computed for each ABR wave. (It is ensured
    via Cross validation, that the regressor that is used for prediction for a specific ABR wave was not trained with this
    specific wave).

    This ABR Threshold detector uses that the sound level can not be predicted below the hearing threshold as
    (per definition of a hearing threshold) there is no information about the stimulus contained in the ABR wave
    below this threshold.
    Therefore in the second step for each Mouse the hearing threshold is detected by dividing the ABR
    curves of a mouse in a set for which the Sound Level can be predicted properly and a set for which the sound
    level cannot be predicted. The sound level that divides these two sets (levels above belong to the first set,
    levels below to the second) is definded as the threshold.

    The second step is performed by fitting for each mouse a piecewise function to the predicted sound level
    vs the true sound level. The piecewise function consists of a constant part for the sound levels below the threshold
    and a polynomial for the sound levels above the threshold.

    The threshold can than either be determined using the breakpoint of the piecewise function, or
    - which is more robust - using the smallest sound level for which the fittet function exeecds a certain threshold
    above the constant part.

    The whole procedure is done for each frequency stimulus separately.

    Parameters
    ----------
    file_name: 'string' or None, defaults to None.
        File name (or path) of a saved model created by saving a trained model using the function save_model_to_file.
        If this is given the model is created from this file and does not need to be trained.

    ---or---

    max_deg : int, defaults to 4.
        Defines the degree of the polynomial part of the piecewise function that is fitted
        to the predicted sound-levels.

    threshold_level : string or float, defaults to 'breakpoint'.
        Defines how the threshold is determined from the fitted piecewise function.
            'breakpoint' - The breakpoint of the piecewise function is returned as the threshold
            float - Value above the constant part that has to be exceeded by the polynmial function.
                    The sound level for which the polynomial part of the piecewise function exeeds this value
                    is returned as the threshold (This is more robust than the 'breakpoint' option as the polynomial part
                    can also approximate a constant function. However the value must be tuned. A value between 2-6 usually works well)

    karwgs_random_forest : dict, defaults to none.
        Keyword arguments passed to the Random Forest that is used as a regressor.
        When 'none' is used the keyword arguments used for the Random Forest are
        {'n_estimators':500,
        'max_features':10,
        'max_samples':0.1,
        'n_jobs':number_of_workers,
        'max_depth':10,
        'random_state':42}

    number_of_workers : int, defaults to 1.
        Number of workers used for parallel executions.

    fourier_cut_off : int, defaults to 100.
        Determines how many fourier modes are kept after preprocessing. Only the largest "fourier_cut_off" frequencies
        are kept.


    Attributes
    ----------
    thresholds_dict_ : dictionary, (key = mouse_id, value = threshold)
        Threshold for each mouse.

    thresholds_per_curve_: numpy-array, shape (data.shape[0],)
        A numpy array that contains for each curve in the dataset that is given to the function
        fit_and_predict_data_set, the corresponding threshold (= the threshold for this mouse).

    """

    def __init__(self, *args, file_name=None, **kwargs):

        # when a file name is given load the estimators from it, else create an empty defaultdict
        if file_name is not None:
            print('Loading model from file ' + file_name)
            with open(file_name, 'rb') as input_file:
                loaded_model = pickle.load(input_file)

            # load the saved objects into the corresponding attributes (the single_frequency_threshold_estimators_
            # have to be converted into a default dict)
            self._ABR_Threshold_Detector_single_stimulus_arguments = loaded_model[
                '_ABR_Threshold_Detector_single_stimulus_arguments']
            args = self._ABR_Threshold_Detector_single_stimulus_arguments['args']
            kwargs = self._ABR_Threshold_Detector_single_stimulus_arguments['kwargs']
            _ABR_Threshold_Detector_single_stimulus_generator_function = lambda: ABR_Threshold_Detector_single_stimulus(
                *args, **kwargs)
            self.single_frequency_threshold_estimators_ = defaultdict(
                _ABR_Threshold_Detector_single_stimulus_generator_function,
                loaded_model['single_frequency_threshold_estimators_'])
            self.thresholds_dict_ = loaded_model['thresholds_dict_']
            print('done.')
        else:
            # create a default dict, that will have keys = frequency of stimulus and values = single stimulus
            # frequency detectors
            self._ABR_Threshold_Detector_single_stimulus_arguments = {'args': args, 'kwargs': kwargs}
            _ABR_Threshold_Detector_single_stimulus_generator_function = lambda: ABR_Threshold_Detector_single_stimulus(
                *args, **kwargs)
            self.single_frequency_threshold_estimators_ = defaultdict(
                _ABR_Threshold_Detector_single_stimulus_generator_function)

    def fit_and_predict_data_set(self, mouse_id, sound_level, frequency, time_series, data=None):
        """
        Preprocesses Data, Trains Model for Sound Level Regression and predicts threshold for data.
        Data can be either given as a pandas data frame or as separate numpy arrays.

        Automatically applies crossvalidation to data using n_splits number of folds. The cross validation
        is a group crossvalidation with the mouse-ids being the groups
        Sound Level Regressors are trained on a subset of the mice and then predict sound levels
        of the left out mice. The predicted sound levels are then used to determine the thresholds
        by fitting a piecewise function.

        This function generates and calls a single stimulus ABR Threshold estimator for each separate
        stimulus frequency that is contained in frequency

        Parameters
        ----------
        mouse_id : string or numpy-array with shape (total number of ABR waves, )
            Vector with mouse_ids of each ABR wave or name of column with mouse_ids in data.

        sound_level : string or numpy-array with shape (total number of ABR waves, )
            Vector with sound levels of each ABR wave or name of column with sound_levels in data.

        frequency : string or numpy-array with shape (total number of ABR waves, )
            Vector with frequency of stimulus belonging to the ABR waves or name of column with frequencies in data.

        time_series : list of strings or numpy-array with shape (total number of ABR waves, number of time steps)
            Either a list of strings, where each string is a columns name in data that contains the ABR value at a certain point in time
            or a numpy array where each row is a time series containing the ABR waves.

        data : pandas-data-frame or None, default None
            A data frame that contains an ABR wave plus metadata in each row.
            Each row contains:
                - an identifier of the mouse given by the column provided in mouse_id
                - the stimulus sound level given by the column provided in sound_level
                - the ABR wave that was recorded for the mouse-sound_level pair.
                The columns of the series are given as a list in time_series.

            If data is given then, mouse_id, sound_level and time_series must provide the names of the columns
            where the data is stored. Otherwise the data must be provided as separate numpy-arrays which match in dimensions.

        Returns
        -------
        thresholds_per_curve_ : numpy-array, shape (total number of ABR waves,)
            A numpy array that contains for each curve in the dataset
            the corresponding threshold (= the threshold for this mouse-frequency pair).
        """

        if type(data) is pd.core.frame.DataFrame:
            self._mouse_id = data.loc[:, mouse_id].to_numpy()
            self._sound_level = data.loc[:, sound_level].to_numpy()
            self._frequency = data.loc[:, frequency].to_numpy()

            _time_series = data.loc[:, time_series].to_numpy()

        else:
            self._mouse_id = mouse_id
            self._sound_level = sound_level
            self._frequency = frequency
            _time_series = time_series

        self._frequencies_unique = np.unique(self._frequency)
        self._mouse_ids_unique = np.unique(self._mouse_id)

        self._number_of_frequencies = len(self._frequencies_unique)
        self._number_of_mice = len(self._mouse_ids_unique)

        self.thresholds_dict_ = {}

        for i, frequency in enumerate(self._frequencies_unique):
            # here we select the part of the data that belongs to a specific stimulus-frequency
            print('Processing stimulus: %s' % frequency)
            frequency_selector = self._frequency == frequency
            mouse_id_freq = self._mouse_id[frequency_selector]
            sound_level_freq = self._sound_level[frequency_selector]
            time_series_freq = _time_series[frequency_selector, :]

            # by asking the default dict for a single_frequency_threshold_estimators it looks up if there is already
            # a single_frequency_threshold_estimator in the dictionary. If not (as it is the
            # case here) then a new one is created.
            ABR_threshold_single_freq = self.single_frequency_threshold_estimators_[frequency]
            # Now the single stimulus threshold estimator is fed with the selected data
            ABR_threshold_single_freq.fit_and_predict_data_set(mouse_id_freq, sound_level_freq, time_series_freq);

            # The Thresholds are stored in a dictionary. keys are frequencies of the stimuli
            # values are dicionaries with (key = mouse_id, value = threshold)
            self.thresholds_dict_[frequency] = ABR_threshold_single_freq.thresholds_dict_

        thresholds_per_curve_ = self.get_thresholds_per_curve(self._frequency, self._mouse_id)

        print('...completed processing the data set.')

        return thresholds_per_curve_

    def get_thresholds_per_curve(self, frequency, mouse_id):
        """
        Helper function that computes a numpy array that contains the hearing threshold for each curve
        (= the hearing threshold of the mouse from which this curve is recorded) from
        the thresholds_dict_, that contains the thresholds for each mouse.
        
        Parameters
        ----------
        mouse_id : string or numpy-array with shape (total number of ABR waves, )
            Vector with mouse_ids of each ABR wave or name of column with mouse_ids in data.

        frequency : string or numpy-array with shape (total number of ABR waves, )
            Vector with frequency of stimulus belonging to the ABR waves or name of column with frequencies in data.
            
        
        Returns
        -------
        thresholds_per_curve_ : numpy-array, shape (total number of ABR waves,)
            A numpy array that contains for each curve in the dataset the corresponding threshold (= the threshold for this mouse).

        """
        thresholds_per_curve_ = np.zeros_like(frequency)

        for freq in np.unique(frequency):
            frequency_selector = frequency == freq
            mouse_id_freq = mouse_id[frequency_selector]

            ABR_threshold_single_freq = self.single_frequency_threshold_estimators_[freq]

            thresholds_per_curve_[frequency_selector] = ABR_threshold_single_freq.get_thresholds_per_curve(
                mouse_id_freq)

        return thresholds_per_curve_

    def predict_new(self, mouse_id, sound_level, frequency, time_series, data=None):
        """
        Gives thresholds for a new data set (training has to be done first).

        Parameters
        ----------
        mouse_id : string or numpy-array with shape (total number of ABR waves, )
            Vector with mouse_ids of each ABR wave or name of column with mouse_ids in data.

        sound_level : string or numpy-array with shape (total number of ABR waves, )
            Vector with sound levels of each ABR wave or name of column with sound_levels in data.

        frequency : string or numpy-array with shape (total number of ABR waves, )
            Vector with frequency of stimulus belonging to the ABR waves or name of column with frequencies in data.

        time_series : list of strings or numpy-array with shape (total number of ABR waves, number of time steps)
            Either a list of strings, where each string is a columns name in data that contains the ABR value at a certain point in time
            or a numpy array where each row is a time series containing the ABR waves.

        data : pandas-data-frame or None, default None
            A data frame that contains an ABR wave plus metadata in each row.
            Each row contains:
                - an identifier of the mouse given by the column provided in mouse_id
                - the stimulus sound level given by the column provided in sound_level
                - the ABR wave that was recorded for the mouse-sound_level pair. The columns of the series are given as a list in time_series.

            If data is given then, mouse_id, sound_level and time_series must provide the names of the columns
            where the data is stored. Otherwise the data must be provided as separate numpy-arrays which match in dimensions.

        Returns
        -------
        thresholds_per_curve_ : numpy-array, shape (total number of ABR waves,)
            A numpy array that contains for each curve in the dataset
            the corresponding threshold (= the threshold for this mouse-frequency pair).
        """
        if type(data) is pd.core.frame.DataFrame:
            self._mouse_id = data.loc[:, mouse_id].to_numpy()
            self._sound_level = data.loc[:, sound_level].to_numpy()
            self._frequency = data.loc[:, frequency].to_numpy()

            _time_series = data.loc[:, time_series].to_numpy()

        else:
            self._mouse_id = mouse_id
            self._sound_level = sound_level
            self._frequency = frequency
            _time_series = time_series

        self._frequencies_unique = np.unique(self._frequency)
        self._mouse_ids_unique = np.unique(self._mouse_id)

        self._number_of_frequencies = len(self._frequencies_unique)
        self._number_of_mice = len(self._mouse_ids_unique)

        self.thresholds_dict_ = {}

        for i, frequency in enumerate(self._frequencies_unique):
            # here we select the part of the data that belongs to a specific stimulus-frequency
            print('Processing stimulus: %s' % frequency)
            frequency_selector = self._frequency == frequency
            mouse_id_freq = self._mouse_id[frequency_selector]
            sound_level_freq = self._sound_level[frequency_selector]
            time_series_freq = _time_series[frequency_selector, :]

            # by asking the default dict for a single_frequency_threshold_estimators it looks up if there is already
            # a single_frequency_threshold_estimator in the dictionary. If not (as it is the
            # case here) then a new one is created.
            ABR_threshold_single_freq = self.single_frequency_threshold_estimators_[frequency]

            # Now the single stimulus threshold estimator is fed with the selected data
            ABR_threshold_single_freq.predict_new(mouse_id_freq, sound_level_freq, time_series_freq);

            # The Thresholds are stored in a dictionary. keys are frequencies of the stimuli
            # values are dicionaries with (key = mouse_id, value = threshold)
            self.thresholds_dict_[frequency] = ABR_threshold_single_freq.thresholds_dict_

        thresholds_per_curve_ = self.get_thresholds_per_curve(self._frequency, self._mouse_id)

        print('...completed processing the data set.')

        return thresholds_per_curve_

    def save_model_to_file(self, file_name="threshold_detector.pkl"):
        """
        Saves trained model to file.

        Parameters
        ----------
        file_name : string
            File name (or path) of to save model to.

        Returns
        -------
        -
        """

        objects_to_save = {}
        objects_to_save['single_frequency_threshold_estimators_'] = dict(self.single_frequency_threshold_estimators_)
        objects_to_save['thresholds_dict_'] = self.thresholds_dict_
        objects_to_save[
            '_ABR_Threshold_Detector_single_stimulus_arguments'] = self._ABR_Threshold_Detector_single_stimulus_arguments

        with open(file_name, "wb") as output_file:
            pickle.dump(objects_to_save, output_file)
        print(f'Model saved to {file_name}')


class ABR_Threshold_Detector_single_stimulus:
    """
    ABR Threshold Detector for a single stimulus.

    The ABR Threshold Detector uses the "Sound Level Regression" method to detect
    the hearing Thresholds in an ABR data set.
    This class is only used for ABR data that results from a single stimulus.


    The Sound Level Regression method works in two steps:
    First a regression model (in this case a Random Forrest) is trained to predict the exact sound
    level of an ABR curve from its wave form (data is preprocessed by converting to power spectrum and cut of at high
    frequencies). Then a prediction is computed for each ABR wave. (It is ensured
    via Cross Validation, that the regressor that is used for prediction for a specific ABR wave was not trained with this
    specific wave).

    This ABR Threshold detector uses that the sound level can not be predicted below the hearing threshold as
    (per definition of a hearing threshold) there is no information about the stimulus contained in the ABR wave
    below this threshold.
    Therefore in the second step for each Mouse the hearing threshold is detected by dividing the ABR
    curves of a mouse in a set for which the Sound Level can be predicted properly and a set for which the sound
    level cannot be predicted. The sound level that divides these two sets (levels above belong to the first set,
    levels below to the second) is defined as the threshold.

    The second step is performed by fitting for each mouse a piecewise function to the predicted sound level
    vs the true sound level. The piecewise function consists of a constant part for the sound levels below the threshold
    and a polynomial for the sound levels above the threshold.

    The threshold can than either be determined using the breakpoint of the piecewise function, or
    - which is more robust - using the smallest sound level for which the fittet function exeecds a certain threshold
    above the constant part.

    The whole procedure is done for each frequency stimulus separately.

    Parameters
    ----------
    max_deg : int, defaults to 4.
        Defines the degree of the polynomial part of the piecewise function that is fitted
        to the predicted sound-levels.

    threshold_level : string or float, defaults to 'breakpoint'.
        Defines how the threshold is determined from the fitted piecewise function.
            'breakpoint' - The breakpoint of the piecewise function is returned as the threshold
            float - Value above the constant part that has to be exceeded by the polynmial function.
                    The sound level for which the polynomial part of the piecewise function exeeds this value
                    is returned as the threshold (This is more robust than the 'breakpoint' option as the polynomial part
                    can also approximate a constant function. However the value must be tuned. A value between 2-6 usually works well)

    karwgs_random_forest : dict, defaults to none.
        Keyword arguments passed to the Random Forest that is used as a regressor.
        When 'none' is used the keyword arguments used for the Random Forest are
        {'n_estimators':500,
        'max_features':10,
        'max_samples':0.1,
        'n_jobs':number_of_workers,
        'max_depth':10,
        'random_state':42}

    number_of_workers : int, defaults to 1.
        Number of workers used for parallel executions.

    fourier_cut_off : int, defaults to 100.
        Determines how many fourier modes are kept after preprocessing. Only the largest "fourier_cut_off" frequencies
        are kept.


    Attributes
    ----------
    thresholds_dict_ : dictionary, (key = mouse_id, value = threshold)
        Threshold for each mouse (only one stimulus). (For multi stimuli use ABR_Threshold_Detector_multi_stimulus)

    """

    def __init__(self, max_deg=4, threshold_level='breakpoint', karwgs_random_forest=None, number_of_workers=1,
                 fourier_cut_off=100):

        if karwgs_random_forest == None:
            karwgs_random_forest = {'n_estimators': 500,
                                    'max_features': 10,
                                    'max_samples': 0.1,
                                    'n_jobs': number_of_workers,
                                    'max_depth': 10,
                                    'random_state': 42}

        self._sound_level_regressor = RandomForestRegressor(**karwgs_random_forest)

        self.number_of_workers = number_of_workers

        self.max_deg = max_deg
        self.threshold_level = threshold_level

        self.fourier_cut_off = fourier_cut_off

    def fit_and_predict_data_set(self, mouse_id, sound_level, time_series, data=None, n_splits=5):
        """
        Preprocesses Data, Trains Model for Sound Level Regression and predicts threshold for data.
        Data can be either given as a pandas data frame or as separate numpy arrays.

        Automatically applies crossvalidation to data using n_splits number of folds. The cross validation
        is a group crossvalidation with the mouse-ids being the groups
        Sound Level Regressors are trained on a subset of the mice and then predict sound levels
        of the left out mice. The predicted sound levels are then used to determine the thresholds
        by fitting a piecewise function.

        Parameters
        ----------
        mouse_id : string or numpy-array with shape (total number of ABR waves, )
            Vector with mouse_ids of each ABR wave or name of column with mouse_ids in data.

        sound_level : string or numpy-array with shape (total number of ABR waves, )
            Vector with sound levels of each ABR wave or name of column with sound_levels in data.

        time_series : list of strings or numpy-array with shape (total number of ABR waves, number of time steps)
            Either a list of strings, where each string is a columns name in data that contains the ABR value at a
            certain point in time
            or a numpy array where each row is a time series containing the ABR waves.

        data : pandas-data-frame or None, default None
            A data frame that contains an ABR wave plus metadata in each row.
            Each row contains:
                - an identifier of the mouse given by the column provided in mouse_id
                - the stimulus sound level given by the column provided in sound_level
                - the ABR wave that was recorded for the mouse-sound_level pair. The columns of the series are given as
                a list in time_series.

            If data is given then, mouse_id, sound_level and time_series must provide the names of the columns
            where the data is stored. Otherwise the data must be provided as separate numpy-arrays which match in dimensions.

        Returns
        -------
        thresholds_per_curve_ : numpy-array, shape (total number of ABR waves,)
            A numpy array that contains for each curve in the dataset the corresponding threshold (= the threshold for this mouse).
        """

        # if data is given in data frame format convert into numpy arrays
        if type(data) is pd.core.frame.DataFrame:
            self._mouse_id = data.loc[:, mouse_id].to_numpy()
            self._sound_level = data.loc[:, sound_level].to_numpy()
            _time_series = data.loc[:, time_series].to_numpy()
        else:
            self._mouse_id = mouse_id
            self._sound_level = sound_level
            _time_series = time_series  # we do not store the time series data, as it is potentially large

        self._mouse_ids_unique = np.unique(self._mouse_id)

        self._number_of_mice = len(self._mouse_ids_unique)

        self.thresholds_dict_ = {}

        # preprocessing:
        # transform timeseries to power spectrum and cut of the first components (as the power spectrum is sparse)
        print('preprocessing time series data...')
        preprocessed_time_series = np.abs(np.fft.rfft(_time_series))[:, :self.fourier_cut_off]

        # the main algorithm:
        # first train and estimate a sound level regressor and then estimate the thresholds from the
        # predicted sound levels
        print('train sound level regressor and compute sound level estimates...')
        self._cv_train_and_estimate_sound_level_regressor(preprocessed_time_series, n_splits=5)
        print('estimate thresholds from sound level estimates...')
        self._estimate_Thresholds_from_predicted_sound_levels()

        # transform data from dictionary that contains threshold for each mouse into a numpy array that contains threshold
        # for each curve
        thresholds_per_curve_ = self.get_thresholds_per_curve(self._mouse_id)

        return thresholds_per_curve_

    def get_thresholds_per_curve(self, mouse_id):
        """
        Helper function that computes a numpy array that contains the hearing threshold for each curve
        (= the hearing threshold of the mouse from which this curve is recorded) from
        the thresholds_dict_, that contains the thresholds for each mouse.
        
        Parameters
        ----------
        mouse_id : string or numpy-array with shape (total number of ABR waves, )
            Vector with mouse_ids of each ABR wave or name of column with mouse_ids in data.
            
        
        Returns
        -------
        tresholds_per_curve_ : numpy-array, shape (total number of ABR waves,)
            A numpy array that contains for each curve in the dataset the corresponding threshold (= the threshold for this mouse).

        """

        thresholds_per_curve_ = np.zeros_like(mouse_id)
        for mid in np.unique(mouse_id):
            selector = mouse_id == mid
            thresholds_per_curve_[selector] = self.thresholds_dict_[mid]
        return thresholds_per_curve_

    def predict_new(self, mouse_id, sound_level, time_series, data=None):
        """
        Gives thresholds for a new data set (training has to be done first).
        
        Parameters
        ----------
        mouse_id : string or numpy-array with shape (total number of ABR waves, )
            Vector with mouse_ids of each ABR wave or name of column with mouse_ids in data.

        sound_level : string or numpy-array with shape (total number of ABR waves, )
            Vector with sound levels of each ABR wave or name of column with mouse_ids in data.
        
        time_series : list of strings or numpy-array with shape (total number of ABR waves, number of time steps)
            Either a list of strings, where each string is a columns name in data that contains the ABR value at a certain point in time
            or a numpy array where each row is a time series containing the ABR waves.

        data : pandas-data-frame or None, default None
            A data frame that contains an ABR wave plus metadata in each row. 
            Each row contains:
                - an identifier of the mouse given by the column provided in mouse_id
                - the stimulus sound level given by the column provided in sound_level
                - the ABR wave that was recorded for the mouse-sound_level pair. The columns of the series are given as a list in time_series.
            
            If data is given then, mouse_id, sound_level and time_series must provide the names of the columns
            where the data is stored. Otherwise the data must be provided as separate numpy-arrays which match in dimensions.
            
        Returns
        -------
        thresholds_per_curve_ : numpy-array, shape (total number of ABR waves,)
            A numpy array that contains for each curve in the dataset the corresponding threshold (= the threshold for this mouse).
        """
        # if data is given in data frame format convert into numpy arrays
        if type(data) is pd.core.frame.DataFrame:
            self._mouse_id = data.loc[:, mouse_id].to_numpy()
            self._sound_level = data.loc[:, sound_level].to_numpy()
            _time_series = data.loc[:, time_series].to_numpy()
        else:
            self._mouse_id = mouse_id
            self._sound_level = sound_level
            _time_series = time_series  # we do not store the time series data, as it is potentially large

        self._mouse_ids_unique = np.unique(self._mouse_id)

        self._number_of_mice = len(self._mouse_ids_unique)

        self.thresholds_dict_ = {}

        # preprocessing:
        # transform timeseries to power spectrum and cut of the first components (as the power spectrum is sparse)
        print('preprocessing time series data...')
        preprocessed_time_series = np.abs(np.fft.rfft(_time_series))[:, :self.fourier_cut_off]
        print('compute sound level estimates...')
        self._sound_levels_predicted = self._sound_level_regressor.predict(preprocessed_time_series)
        print('estimate thresholds from sound level estimates...')
        self._estimate_Thresholds_from_predicted_sound_levels()

        # transform data from dictionary that contains threshold for each mouse into a numpy array that contains threshold
        # for each curve
        thresholds_per_curve_ = self.get_thresholds_per_curve(self._mouse_id)

        return thresholds_per_curve_

    def _cv_train_and_estimate_sound_level_regressor(self, preprocessed_time_series, n_splits=5):
        """
        Computes predicted sound levels.
        This function splits the data into n_splits using GroupKFold with the mouse ids determining the groups.
        Then for each split we have a training and a test set. A random forrest regressor is trained
        on the training set to predict the sound levels of each curve using the preprocessed time series as
        input. The learned regressor is then applied to the test set and the predicted sound levels are stored.
        
        Parameters
        ----------
        preprocessed_time_series : ndarray of shape (total number of ABR waves, fourier_cut_off)
            Preprocessed ABR waves.
            
        n_splits : int, defaults to 5
            Number of splits for the GroupKFold.
            
        Returns
        -------
        -
        """
        sound_levels = self._sound_level
        mouse_id = self._mouse_id
        sound_level_regressor = self._sound_level_regressor

        sound_levels_predicted = np.ones_like(sound_levels)

        kf = GroupKFold(n_splits=n_splits)
        cross_validated_split = kf.split(preprocessed_time_series, sound_levels, groups=mouse_id)

        test_errors = []

        # cross validated training and estimation on the data set

        for train_index, estimation_index in cross_validated_split:
            preprocessed_time_series_training = preprocessed_time_series[train_index]
            preprocessed_time_series_estimation = preprocessed_time_series[estimation_index]

            sound_levels_training = sound_levels[train_index]
            sound_levels_estimation = sound_levels[estimation_index]

            sound_level_regressor.fit(preprocessed_time_series_training, sound_levels_training)
            sound_levels_predicted[estimation_index] = sound_level_regressor.predict(
                preprocessed_time_series_estimation)

            estimation_error = mean_squared_error(sound_levels_predicted[estimation_index], sound_levels_estimation,
                                                  squared=False)

        # train sound_level regressor on entire data set
        sound_level_regressor.fit(preprocessed_time_series, sound_levels)

        self._sound_levels_predicted = sound_levels_predicted

    def _estimate_Thresholds_from_predicted_sound_levels(self):
        """
        Computes Thresholds for each mouse and stores it in a thresholds_dict_.
        Estimates Thresholds by fitting a piecewise function to the predicted sound levels.
        Uses multithreading to perform a parallel loop over the different mice and estimate the threshold
        for each mouse.
        
        Parameters
        ----------
        -
            
        Returns
        -------
        -
        """

        sound_levels = self._sound_level
        mouse_id = self._mouse_id
        sound_levels_predicted = self._sound_levels_predicted
        number_of_mice = self._number_of_mice

        all_mice = self._mouse_ids_unique

        # Loop over all mice in parallel. To be able to show progressbar
        # this is implemented such, that in a normal for-loop we iterate over all mice
        # and pack the data for these mice into lists until we have as many mice as we have parallel workers.
        # Then the actual threshold estimation is executed in parallel.
        with multiprocessing.Pool(self.number_of_workers) as pool:

            mids = []
            data_for_single_mice = []
            for i in progressbar.progressbar(range(number_of_mice)):

                # collect data for each mouse and pack it into a list
                mid = all_mice[i]
                selector_mouse_i = (mouse_id == mid)
                mids.append(mid)
                data_for_single_mice.append((sound_levels[selector_mouse_i], sound_levels_predicted[selector_mouse_i]))

                # when we have as many mice in the list as we have workers do the actual processing in parallel:
                if len(data_for_single_mice) == self.number_of_workers:
                    # do parallel threshold estimation (this is quite slow, therefore it is parallelized)
                    # print(data_for_single_mice)
                    threshold_list = pool.map(self._estimate_threshold_for_single_mouse, data_for_single_mice)

                    # collect the results in the global dictionary
                    self.thresholds_dict_.update(zip(mids, threshold_list))

                    # empty the lists as the mice have been processed allready, now we can start collecting a
                    # new batch of mice to process in parallel.
                    mids = []
                    data_for_single_mice = []

            # Process leftovers:
            # Look if there are still unprocesses mice in the data_for_single_mice list and process them.
            if len(data_for_single_mice) > 0:
                threshold_list = pool.map(self._estimate_threshold_for_single_mouse, data_for_single_mice)

                self.thresholds_dict_.update(zip(mids, threshold_list))

    def _estimate_threshold_for_single_mouse(self, data):
        """
        Threshold estimator for a single mouse
        Fits a piecewise function (constant, polymial) to data x,y and determines a threshold in x when
        the function f: x->y starts deviating from a constant.
        
        Has two options for finding the threshold controlled by self.threshold_level: 
        either it declares the breakpoint of the piecewise function as the threshold,
        or the first value in x for which f: x->y is deviating more than self.threshold_level from the constant
        part of the piece wise function.
        The later option is more robust to a nonlinear function that fits a constant function after the breakpoint
        and starts rising only after a while.
        
        Parameters
        ----------
        data :  list, shape 2
            list containing the true sound levels (x) and the predicted sound levels(y)
            
        Returns
        -------
        threshold : int
            threshold where f: x->y starts deviating form a constant
        """

        x = data[0]
        y = data[1]

        # print('#### x ####')
        # print(x)
        # print()

        # print('#### y ####')
        # print(y)
        # print()

        max_deg = self.max_deg
        threshold_level = self.threshold_level

        if len(x) <= 2:

            threshold = 999
            # if there are only two sound levels or less proper estimation can not be done and we default to 999
            # which means that the mouse cannot hear

        else:
            # print('#### len(x) ####')
            # print(len(x))
            ths_best, y_estim_best = _loop_through_breakpoints_and_fit_piecewiese_functions(x, y, max_deg,
                                                                                            k_fold=len(x))

            if threshold_level == 'breakpoint':
                threshold = x - np.mod(x, 5) + 5  # round to next value divisible by 5
            else:
                # if a specific 'threshold_level' is given then take
                # the first point right of the breakpoint of the piecewise function
                # that surpases constant value + threshold_level

                candidates = (y_estim_best - y_estim_best[x.argmin()])
                thresholded_candidates_y = candidates[candidates >= threshold_level]
                thresholded_candidates_x = x[candidates > threshold_level]
                if len(thresholded_candidates_y) == 0:
                    threshold = 999
                else:
                    threshold = thresholded_candidates_x[thresholded_candidates_y.argmin()]

            if x.max() < threshold:
                threshold = 999

            if x.min() > threshold:
                threshold = max(5, x.min() - 5)

        return threshold
