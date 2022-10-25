# Applied Deep Learning - Computer Vision - Image Classification

## RSNA-MICCAI Brain Tumor Radiogenomic Classification

A malignant tumor in the brain is a life-threatening condition. Known as glioblastoma, it's both the most common form of brain cancer in adults and the one with the worst prognosis, with median survival being less than a year. The presence of a specific genetic sequence in the tumor known as MGMT promoter methylation has been shown to be a favorable prognostic factor and a strong predictor of responsiveness to chemotherapy.

Currently, genetic analysis of cancer requires surgery to extract a tissue sample. Then it can take several weeks to determine the genetic characterization of the tumor. Depending upon the results and type of initial therapy chosen, a subsequent surgery may be necessary. If an accurate method to predict the genetics of the cancer through imaging (i.e., radiogenomics) alone could be developed, this would potentially minimize the number of surgeries and refine the type of therapy required.


### Project Idea & Approach

This project is focused on the building or re-implementing neural network architecture that operates on an existing dataset that is already publicly available. Since the project idea was inspired from already closed Kaggle competition, we will try to re-implement some of the most prominent shared notebooks neural networks (including the 1st place Kaggle competition model, which uses 3D CNN), and even to build our own neural network in order to imporve the state of art. 


### Dataset Description

The data is defined by three cohorts: Training, Validation, and Testing. These 3 cohorts are structured as follows: Each independent case has a dedicated folder identified by a five-digit number. Within each of these “case” folders, there are four sub-folders, each of them corresponding to each of the structural multi-parametric MRI (mpMRI) scans, in DICOM format. The exact mpMRI scans included are:

-   Fluid Attenuated Inversion Recovery (FLAIR)
-   T1-weighted pre-contrast (T1w)
-   T1-weighted post-contrast (T1Gd)
-   T2-weighted (T2)

#### Additional Information

- **Files**: 400116 files
- **Size**: 136.85 GB
- **Type**: dcm, csv


### Work-breakdown structure


### References

#### Scientific Papers

##### A Deep Learning Approach for Brain Tumor Classification and Segmentation Using a Multiscale Convolutional Neural Network
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7912940/


##### Aleatoric uncertainty estimation with test-time augmentation for medical image segmentation with convolutional neural networks
https://www.sciencedirect.com/science/article/pii/S0925231219301961


##### Brain Tumor Classification Using Deep Learning Technique - A Comparison between Cropped, Uncropped, and Segmented Lesion Images with Different Sizes

https://www.researchgate.net/publication/338540693_Brain_Tumor_Classification_Using_Deep_Learning_Technique_-_A_Comparison_between_Cropped_Uncropped_and_Segmented_Lesion_Images_with_Different_Sizes


##### EfficientNetV2: Smaller Models and Faster Training
https://arxiv.org/abs/2104.00298

##### Interpreting Deep Neural Networks for Medical Imaging using Concept Graphs
https://paperswithcode.com/paper/abstracting-deep-neural-networks-into-concept


#### Other
##### Magnetic Resonance Imaging (MRI) of the Brain and Spine: Basics
https://case.edu/med/neurology/NR/MRI%20Basics.htm

##### A Comprehensive Guide to Convolutional Neural Networks
https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53


##### RSNA-MICCAI Brain Tumor Radiogenomic Classification
https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification/overview

##### Kaggle - 1st place solution with very simple code
https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification/discussion/281347


##### Kaggle Competition - RSNA-MICCAI Brain Tumor Radiogenomic Classification
https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification/overview
