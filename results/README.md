# What is this?

This folder is the location where 

+ the files containing the automatically determined hearing thresholds, 
+ but also the results of the Jupyter notebooks are stored.

The complete results can be found at [https://zenodo.org/deposit/5779876](https://zenodo.org/deposit/5779876).

## Threshold predictions by neural networks

The csv files

+ `./GMC_data_GMCtrained_NN_predictions.csv`, 
+ `./GMC_data_INGtrained_NN_predictions.csv`, 
+ `./ING_data_GMCtrained_NN_predictions.csv`, 
+ `./ING_data_INGtrained_NN_predictions.csv`

store the predictions made by GMC or ING trained neural networks on GMC or ING data as follows: 

+ **mouse_id**: mouse identifier 
+ **frequency**: stimulus frequency (Hz) or broadband stimulus ('click' or 100) 
+ **threshold**: manually assessed hearing threshold 
+ **nn_predicted_thr**: hearing threshold predicted by neural networks
+ **nn_predict_probs**: the probability with which the threshold value was predicted  
+ **nn_thr_score**: the **Model II** score of the manually determined threshold value
+ **sl0**, **sl5**, **sl10**, **sl15**, **sl20**, **sl25**, **sl30**, **sl35**, **sl40**, **sl45**, **sl50**, **sl55**, **sl60**, **sl65**, **sl70**, **sl75**, **sl80**, **sl85**, **sl90**, **sl95**: the predictions of **Model I** on the sound pressure levels that feed the second neural network in the second step

## Threshold estimates based on the SLR method

The csv files

+ `./GMC_data_GMCcalibrated_SLR_estimations.csv`, 
+ `./GMC_data_INGcalibrated_SLR_estimations.csv`, 
+ `./ING_data_GMCcalibrated_SLR_estimations.csv`, 
+ `./ING_data_INGcalibrated_SLR_estimations.csv`

store threshold estimates for GMC or ING data based on the SLR method calibrated on GMC or ING data as follows:

+ **mouse_id**: mouse identifier
+ **frequency**: stimulus frequency (Hz) or broadband stimulus ('click' or 100)
+ **threshold**: manually assessed hearing threshold
+ **slr_estimated_thr**: hearing threshold estimated by the SLR method

## Storage locations for result plots

+ The Jupyter notebook `../notebooks/Analysis_of_hearing_curves.ipynb` uses the folders `./hearing_curve_analysis` and `./volcano_plots` to save the results.
+ The Jupyter notebook `../notebooks/Statistical_data_comparison.ipynb` uses `./effect_plots` to save the results.

