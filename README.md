<h2><u>Introduction</u></h2>

This repository contains the code for my master's thesis, titled: **"Optimized Prompting in SAM for Few-Shot and Weakly Supervised Segmentation of Complex Structures and Small Lesions"**, which as accepted at the **MICCAI** conference. In this project, we fine-tune the decoder part of the SAM (Segment Anything Model) using the Vanilla method, where all parameters are updated during training, as depicted in Figure 1. 

Our study explores prompt-guided strategies in SAM for medical image segmentation under few-shot and weakly supervised scenarios. We assess various strategies—such as bounding boxes, positive points, negative points, and their combinations—to optimize the segmentation process.

![Figure 1/ SAM Setup](Figure1.png)

<h2><u>Repository Structure</u></h2>

**1. 'main.py' located indide '_ main _' folder:**

   **Purpose:**
  
  This is the main script where the SAM model is defined and the training process is executed. The 'SAMDataset' function, which is critical for loading and preprocessing the dataset, is included here.
   
  **Key Points:**
  
  - Handles the setup for training and evaluation of the SAM model.
  - Includes the function 'SAMDataset', which you need to configure for your specific dataset.
  - Handles the prompts you wish to use: these prompts are provided as inputs to the model. If there are prompts you do not want to use, you should set them to `None`.
    
  ```python
    outputs = model(pixel_values=batch["pixel_values"].to(device),
                    input_boxes=batch["input_boxes"].to(device),# Use this prompt
                    input_points=None,# Other prompts set to None
                    input_labels=None, # Other prompts set to None
                    multimask_output=False)  # set to 'True' if you want multi-mask output
  ``` 
  **Note:** 'input_points' and 'input_labels' prompts should be used together 

**2. 'dataset.py' located inside 'preprocessing' folder:** 

 **Purpose:**

This file provides a template for setting up your dataset, including necessary transformations and preprocessing steps tailored to your data. 
You don't need to run the 'dataset.py' file directly; the 'SAMDataset' function is invoked within the 'main.py' file during training.

**Key Focus Areas:**

- 'ScaleIntensityRanged' Transform:
   - For images: Adjust the 'a_min' and 'a_max' parameters to match the intensity range of your dataset.
     
   - For labels: Set the 'a_min' and 'a_max' parameters according to the intensity range of your labels.
     If your labels are already binarized (values of 0 and 1), you can skip this transformation for labels.
    
-  Handling of Prompts Functions.

**Note:** For detailed guidance on setting up your own dataset and configuring the prompts functions, please refer to the 'README.md' file located in the 'preprocessing' folder.

**3. 'train.py':** 

 **Purpose:**
 
This file is used to execute the main training script and run the training process.

**4. 're-train.py':** 

 **Purpose:**

This file allows you to continue training from a specific epoch by loading the previous model checkpoints.

**4. 'inference.py':** 

 **Purpose:**

 This file is used to perform testing using the test set.
 
**5. 'requirements.py':** 

 **Purpose:**

Lists all the Python dependencies required to run this project.

Ensure you install all the required packages using the following command:

```python
pip install -r requirements.txt
```

<h2><u>Quick Start</u></h2>

**Install**

- Step 1:

```python
pip install -q monai
pip install -q git+https://github.com/huggingface/transformers.git
pip install -r requirements.txt
```

- Step 2: Download the SAM weights from [SAM repository](https://github.com/facebookresearch/segment-anything#model-checkpoints)

**Train and Test**

- Step 1: Modify the prompting function, select the prompts you wish to use, and adjust the preprocessing steps for ''SAMDataset' in the 'main.py' file.

- Step 2: To execute the training, run the 'train.py' file. If you wish to resume training from a specific epoch, run the 're-train.py' file instead.
  Before doing so, make sure to modify the paths to match your setup.
  
- Step 3: To perform testing, run the 'inference.py' file. Be sure to update the paths to match your setup before executing it.

<h2><u>Citation</u></h2>
Please cite our paper if you use our code or reference our work:

@inproceedings{Siblini2024SAM,
  title={Optimal Prompting in SAM for Few-Shot and Weakly Supervised Medical Image Segmentation},
  author={Siblini, Lara and Andrade-Miranda, Gustavo and Taguelmimt, Kamilia and Visvkis, Dimitris and Bert, Julien},
  booktitle={MICCAI 2024 2nd International Workshop on Foundation Models for General Medical AI. Accepted on July 15},
  year={2024},
}
