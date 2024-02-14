# EEG-Based Motor Imagery Classification Study

This repository contains all the notebooks developed and run on [Kaggle](https://www.kaggle.com/) for the study on EEG-based Motor Imagery Classification.

## Repository Structure

The structure of this repository reflects the three main objectives of my thesis:

**"Regularized Gaussian Functional Connectivity Network with Post-Hoc Interpretation for Improved EEG-based Motor Imagery-BCI Classification."**

The thesis was authored by Daniel Guillermo Garcia Murillo as part of his Ph.D. requirements. [github_link](https://github.com/dggarciam)

## Util Folder

The 'Utils' folder includes the essential functions required for each notebook. Note that the path in the Kaggle directory appended with "/kaggle/input/mi-eeg-classmeth/" refers to the 'Utils' folder in this repository.

## Dependencies

This work uses several repositories developed by the Signal Processing and Recognition Group (SPRG). These can be found in the following links:

- [Visualizations](https://github.com/UN-GCPDS/python-gcpds.visualizations) library for creating topoplots and connectomes illustrations.
- [Deep Learning Architectures](https://github.com/UN-GCPDS/python-gcpds.EEG_Tensorflow_models) library used importing DL architectures compared in this study.
- [Databases Manager](https://github.com/UN-GCPDS/python-gcpds.databases) library to handle all databases used in this study.
- [tf_keras_viz_mod](https://github.com/UN-GCPDS/tf-keras-vis), a fork modified from the original [tf-keras-viz](https://github.com/keisen/tf-keras-vis) repository.

## Data

The data used in this study can be found in the following Kaggle datasets:

- [BCI2a Kaggle Dataset](https://www.kaggle.com/datasets/dggarciam94/bciiv2a-gcpds)
- [GigaScience Kaggle Dataset](https://www.kaggle.com/datasets/dggarciam94/giga-science-gcpds)

## Publications

You can check out more details in the following published articles:

1) García-Murillo D.G., Alvarez-Meza A., Castellanos-Dominguez G. (2021). Single-Trial Kernel-Based Functional Connectivity for Enhanced Feature Extraction in Motor-Related Tasks. Sensors. 2021; 21(8):2750. [https://doi.org/10.3390/s21082750](https://doi.org/10.3390/s21082750) 
2) García-Murillo D.G., Álvarez-Meza A.M., Castellanos-Dominguez C.G. (2023). KCS-FCnet: Kernel Cross-Spectral Functional Connectivity Network for EEG-Based Motor Imagery Classification. Diagnostics. 2023; 13(6):1122. [https://doi.org/10.3390/diagnostics13061122](https://doi.org/10.3390/diagnostics13061122)

## Performance Visualization

![Comparison of DL Architectures](https://github.com/dggarciam/PhD_code_reposotory/assets/25867952/1c507687-74bb-489a-9c98-18d66e1be58f)

The figure above demonstrates the performance and number of parameters in a comparison plot. This shows that our proposals outperform state-of-the-art end-to-end deep learning architectures in accuracy, and they rank second in terms of the number of parameters.
