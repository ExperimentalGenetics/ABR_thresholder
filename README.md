# ABR\_Threshold_Detection

## What is this?

This code can be used to automatically determine hearing thresholds from ABR hearing curves. 

One of the following methods can be used for this purpose:
 
+ neural network (NN) training, 
+ calibration of a self-supervised sound level regression (SLR) method 

on given data sets with manually determined hearing thresholds.

## Installation:

Run inside the [src](./src) directory:

### Installation as python package

```
pip install -e ./src        (Installation as python package)
```

### Installation as conda virtual environment
```
conda create -n abr_threshold_detection python=3.7
conda activate abr_threshold_detection
conda install pip
pip install -e ./src
```

## Usage:
Data files can be downloaded here: [https://zenodo.org/deposit/5779876](https://zenodo.org/deposit/5779876).

For the Jupyter Notebooks (see the [`notebooks`](./notebooks) directory) to run, the path to the data has to be defined. For this, see the corresponding documentation of the respective notebooks.

### Using NNs (`./src/ABR_ThresholdFinder_NN`)

The neural network models were trained in `./src/notebooks/GMCtrained_NN*_training.ipynb` with GMC data and in `./src/notebooks/INGtrained_NN*_training.ipynb` with ING data.

```
import ABR_ThresholdFinder_NN.data_preparation as dataprep
from ABR_ThresholdFinder_NN.models import create_model_1, compile_model_1
```
For automatic threshold detection based on NNs, `GMCtrained_NN_threshold_detection.ipynb` and `INGtrained_NN_threshold_detection.ipynb` in `./src/notebooks` can be used.

```
import ABR_ThresholdFinder_NN.data_preparation as dataprep
import ABR_ThresholdFinder_NN.thresholder as abrthr
```

### Using the SLR method (`./src/ABR_ThresholdFinder_SLR`)

In `./src/notebooks/GMCcalibrated_SLR_threshold_detection.ipynb` and `./src/notebooks/INGcalibrated_SLR_threshold_detection.ipynb` it is shown how to use the module to:

+ train a threshold detector on a data set and estimate the thresholds
+ save a trained model
+ load a model
+ apply a trained threshold estimator to a data set
+ evaluate thresholds by comparing it to a ground truth
+ evaluate thresholds by analysing signal averages

```
import pandas as pd
import numpy as np

from ABR_ThresholdFinder_SLR import ABR_Threshold_Detector_multi_stimulus
from ABR_ThresholdFinder_SLR.evaluations import evaluate_classification_against_ground_truth, plot_evaluation_curve_for_specific_stimulus
```

##### Evaluate thresholds by comparing it with a 'ground truth' (a human set threshold in this case)

For example:

```
# 5dB buffer
evaluation = evaluate_classification_against_ground_truth(GMC_data2, 5, 
                                 frequency = 'frequency',
                                 mouse_id = 'mouse_id',
                                 sound_level = 'sound_level',
                                 threshold_estimated = 'slr_estimated_thr',
                                 threshold_ground_truth = 'threshold')
```     
### Compute and plot evaluation curves that allow to judge the quality of a thresholding

Four threshold types are evaluated and compared:

+ the threshols predicted with neural networks ('threshold NN')
+ the thresholds estimated by a sound level regression method ('threshold SLR')
+ the human ground truth ('threshold manual')
+ a constant threshold ('50')

For more details, please see `Evaluation_of_ML_detected_thresholds.ipynb` in `./src/notebooks`.

## Folder structure:

### [`data`](./data)
Contains the preprocessed ABR and mouse phenotyping datasets from GMC and Ingham et al. in csv format, as well as the mouse ID distributions stored as numpy arrays for neural networks training, validation and testing.

### [`models`](./models)
Contains the trained models of the two neural networks and the SLR method, but also the predictions of the first neural network with which the second neural network was fed.

### [`models_cross-validation`](./models_cross-validation)
Contains the models that resulted from the cross-validation of the neural networks.

### [`notebooks`](./notebooks)
Contains the Jupyter notebooks used for training, testing and evaluation of the neural networks and the SLR method, as well as those used for the hearing curve analysis.

### [`notebooks_reports`](./notebooks_reports)
Contains the contents of Jupyter notebooks in html format.

### [`results`](./results)
Contains the predictions or estimates made by the neural networks or the SLR method for the two data sets from GMC and Ingham et al. but also all the plots made to analyse the results.

### [`src`](./src)
Contains the Python scripts used in the Jupyter notebooks.