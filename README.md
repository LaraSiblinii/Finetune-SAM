**Introduction**

This repository contains the code for our project: **Optimized Prompting in SAM for Few-Shot and Weakly Supervised Segmentation of Complex Structures and Small Lesions**. In this project, we fine-tune the decoder part of the SAM (Segment Anything Model) using the Vanilla method, where all parameters are updated during training, as depicted in Figure 1. 

![Figure 1/ SAM Setup](Figure1.png)

**Repository Structure**

**1. 'main.py'**

 **Purpose:**

This is the main script where the SAM model is defined and the training process is executed. The 'SAMDataset' function, which is critical for loading and preprocessing the dataset, is included here.
 
**Key Points:**

- Handles the setup for training and evaluation of the SAM model. 
- Includes the function 'SAMDataset', which you need to configure for your specific dataset.
