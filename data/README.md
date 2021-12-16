# What is this?

Both GMC and ING data sets should be stored in this folder. 

The entire data sets can be downloaded at [https://zenodo.org/deposit/5779876](https://zenodo.org/deposit/5779876).
## GMC data

### `./GMC/GMC_abr_curves.csv`

Contains the time steps for the ABR curves measured in the GMC. The following data is available for each mouse identifier: 

+ **mouse_id**: mouse identifier
+ **frequency**: stimulus frequency in Hz with the exception of broadband stimulus ('click' or 100)
+ **sound_level**: sound pressure level
+ **threshold**: manually assessed threshold 
+ **validated**: if true, the hearing threshold has been re-validated by ABR-trained users  
+ **still_hears**: if true, the ABR curve is above the assigned threshold value
+ **t0**-**t1952**: 1953 time steps that define an ABR curve

### `./GMC/GMC_mouse_data.csv`

Stores mouse phenotyping data like:

+ **mouse_id**: mouse identifier
+ **mouse_sex**: mouse sex
+ **mouse_line**: mutant mouse line
+ **cohort_id**: identifier of the cohort in which the mouse was measured
+ **cohort_type**: determines whether mutant or control animals
+ **cohort_zygosity**: cohort zygosity (hom/het/hemi)
+ **reference_cohort**: identifier for the cohort with the matched reference control animals for a mutant cohort
+ **gene_symbol**: official symbol of the knocked-out gene
+ **gene_ID**: MGI gene identifier
+ **exp_date**: experiment date

### `./GMC/GMC_mice_with_missing_ref_cohorts.npy`

List of identifiers of mutant mice with reference control cohorts that are not available in the data set since cohorts were not yet completed at the time the data set was created. 

### `./GMC/GMC_train_mice.npy`

List of GMC mouse identifiers used for training the models.

### `./GMC/GMC_valid_mice.npy`

List of GMC mouse identifiers used for validating the models.

### `./GMC/GMC_test_mice.npy`

List of GMC mouse identifiers used for testing the models.

### `./GMC/GMC_train_mice4cross_validation.npy`

List of GMC mouse identifiers used for the cross-validation trainings.

## ING data

### `./ING/ING_abr_curves.csv`

Contains the time steps for the ABR curves measured by Ingham et al (doi: 10.5061/DRYAD.CV803RV). The following data is available for each mouse identifier: 

+ **mouse_id**: mouse identifier
+ **frequency**: stimulus frequency in Hz with the exception of broadband stimulus ('click' and 100)
+ **sound_level**: sound pressure level
+ **threshold**: manually assessed hearing threshold
+ **t0**-**t1952**: 1953 time steps that define an ABR curve

### `./ING/ING_mouse_data.csv`

Stores mouse phenotyping data and manually assessed thresholds in the following columns: 

+ **Colony Prefix**
+ **Mouse Barcode**
+ **Mouse Name**
+ **Pipeline**
+ **Birth Date**
+ **Test Date**
+ **Age at Test**
+ **Genotype**
+ **Genetic Background**
+ **Click Threshold**
+ **6kHz Threshold**
+ **12kHz Threshold**
+ **18kHz Threshold**
+ **24kHz Threshold**
+ **30kHz Threshold**
+ **cohort_type**
+ **mouse_id**

### `./ING/ING_train_mice.npy`

List of ING mouse identifiers used for training the models.

### `./ING/ING_valid_mice.npy`

List of ING mouse identifiers used for validating the models.

### `./ING/ING_test_mice.npy`

List of ING mouse identifiers used for testing the models.

## File `./data4hearing_curves_analysis.csv`

The file combines both the mouse phenotyping data and the manually and automatically determined thresholds for mice from both data sets (GMC and ING).</br>
The following data is available for each mouse identifier:

+ **mouse_id**: mouse identifier
+ **sex**: mouse sex (only available for GMC data)
+ **cohort_type**: determines whether mutant or control animals
+ **cohort_id**: cohort identifier (only available for GMC data)
+ **reference_cohort**: identifier of the cohort with the matched reference control animals for a mutant cohort (only available for GMC data)
+ **gene**: gene knocked out
+ **exp_date**: experiment date
+ **source**: which data set the mouse origins from (GMC or ING)
+ **frequency**: stimulus frequency in Hz with the exception of broadband stimulus ('click' or 100)
+ **th_manual**: manually assessed hearing threshold
+ **th_NN_GMCtrained**: hearing threshold predicted by NNs trained on GMC data
+ **th_NN_INGtrained**: hearing threshold predicted by NNs trained on ING data
+ **th_SLR_GMCcalibrated**: hearing threshold estimated by the SLR method calibrated on GMC data
+ **th_SLR_INGcalibrated**: hearing threshold estimated by the SLR method calibrated on ING data
+ **stimulation**: click, 6, 12, 18, 24 or 30 kHz.