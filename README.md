# Folder structure

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