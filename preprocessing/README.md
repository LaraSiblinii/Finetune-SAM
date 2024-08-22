**Overview**
The *'dataset.py'* file provides a template for setting up a dataset in your project. It includes transformations and preprocessing steps
tailored to your data. This README explains how to customize the *'dataset.py'* file for different datasets, with a particular focus on the
*'ScaleIntensityRanged'* transform and handling of *'prompts'*.

**Note:** The SAMDataset function is defined in *'main.py'*. This README focuses on explaining how to modify *'dataset.py'* but does not 
require running this file directly during training.

**Key Transformations:**
**1. 'ScaleIntensityRanged'**
The *'ScaleIntensityRanged'* transform is used to scale the intensity values of your images and labels. This normalization ensures that the input
data is consistent and suitable for model training. Here’s how to configure it based on your dataset:
**For Images:**
