# List of Jupyter notebooks used in this project

###  `./ING_data_preparation.ipynb`
This notebook is used to create a data set for Deep Learning from a list of archived ABR raw data files 
provided by [Ingham et. al](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000194) and available at [Dryad](https://datadryad.org/stash/dataset/doi:10.5061/dryad.cv803rv).

### `./INGtrained_NN1_training.ipynb`
This notebook is used to train the first convolutional neural network (Model I) with ABR row data provided by [Ingham et. al](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000194). </br>
Model I is trained as classifier to predict if an ABR response is present or not in a single stimulus curve (one frequency, one sound pressure level).
  
### `./INGtrained_NN2_training.ipynb`
This notebook ist used to train the second convolutional neural network (Model II) based on the ABR row data provided by [Ingham et. al](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000194). </br>
Model II is trained as classifier for every stimulus to predict the hearing threshold using the respective class score outputs of Model I as input and the original hearing thresholds as labels.
  
### `./GMCtrained_NN1_training.ipynb`
This notebook is used to train Model I with data provided by the [German Mouse Clinic](https://www.mouseclinic.de/). </br>
Model I is trained as a classifier to predict if an ABR response is present or not in a single stimulus curve (one frequency, one sound pressure level).
   
### `./GMCtrained_NN2_training.ipynb`
This notebook is used to train Model II based on data provided by the [German Mouse Clinic](https://www.mouseclinic.de/). </br>
Model II is trained as a classifier for every stimulus to predict the hearing threshold using the respective class score outputs of Model I 
as input and the original hearing thresholds as labels.

### `./GMCtrained_NN1_cross_validation.ipynb`
In this notebook, a five-fold grouped cross-validation was performed for Model I with data from the [German Mouse Clinic](https://www.mouseclinic.de/), where the groups consist of mice.

### `./GMCtrained_NN2_cross_validation.ipynb`
In this notebook, a five-fold grouped cross-validation was performed for Model II with data from the [German Mouse Clinic](https://www.mouseclinic.de/), where the groups consist of mice. </br>
The class score outputs of Model I are used as input for the second model.

### `./GMCtrained_NN_threshold_detection.ipynb`
This notebook is used to detect ABR hearing thresholds using neural networks (NN) trained on ABR data from the [German Mouse Clinic](https://www.mouseclinic.de/).</br> 
The threshold detection is done on ABR hearing curves from the [German Mouse Clinic](https://www.mouseclinic.de/) (GMC data) as well as on ABR hearing curves provided by [Ingham et. al](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000194) (ING data).

### `./INGtrained_NN_threshold_detection.ipynb`
This notebook is used to detect ABR hearing thresholds using neural networks (NN) trained on ABR data provided by [Ingham et. al](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000194).</br> 
The threshold detection is done on ABR hearing curves provided by [Ingham et. al](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000194) (ING data) as well as on ABR hearing curves from the [German Mouse Clinic](https://www.mouseclinic.de/) (GMC data).

### `./GMCcalibrated_SLR_threshold_detection.ipynb`
This notebook is used for ABR hearing threshold detection and evaluation using a sound level regression (SLR) method.</br> 
The method is calibrated here on the [GMC](https://www.mouseclinic.de/) data set used for training the convolutional neural networks. 
The model is then applied to both GMC and ING data.

### `./INGcalibrated_SLR_threshold_detection.ipynb`
This notebook is used for ABR hearing threshold detection and evaluation using a sound level regression (SLR) method.</br>
The method is calibrated here on the [Ingham et. al](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000194) data set used for training the convolutional neural networks.
The model is then applied to both GMC and ING data.

### `./Evaluation_of_ML_detected_thresholds.ipynb`
This notebook is used to evaluate the hearing thresholds predicted by neural networks (NN) or estimated by a sound level regression method (SLR) by comparison with a ground truth which are manually assessed thresholds. 

### `./Analysis_of_hearing_curves.ipynb`
In this notebook, raw data was subjected to NN and SLR threshold finding for all mice in both datasets ([GMC](https://www.mouseclinic.de/) and [ING](https://journals.plos.org/plosbiology)). </br>
In a first step, all thresholds, both those determined manually and those determined automatically, are combined into a single data set.</br>
Hearing curves are then generated for all mice in the data set to compare the differences between the hearing curves of mutants and controls using the three methods (manual, NN, SLR).

### `./Statistical_data_comparison.ipynb`
In this notebook, the GMC mouse lines are analysed and a list of genes with significant (Wilcoxon test) strong effects (Cliff's delta) is identified.