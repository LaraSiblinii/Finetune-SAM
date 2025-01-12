<h2>

<u>Introduction</u>

</h2>

This repository contains the code for my master's thesis, titled: **"Optimized Prompting in SAM for Few-Shot and Weakly Supervised Segmentation of Complex Structures and Small Lesions"**, which as accepted at the **MICCAI 2024 2nd International Workshop on Foundation Models for General Medical AI**. In this project, we fine-tune the decoder part of the SAM (Segment Anything Model) using the Vanilla method, where all parameters are updated during training, as depicted in Figure 1.

Our study explores prompt-guided strategies in SAM for medical image segmentation under few-shot and weakly supervised scenarios. We assess various strategies---such as bounding boxes, positive points, negative points, and their combinations---to optimize the segmentation process.

![Figure 1/ SAM Setup](Figure1.png)

<h2>

<u>Repository Structure</u>

</h2>

**1. configs folder:**

Contains an example of a configuration file (`debug.yaml`) used for training and performing inference with different types of prompts. The prompts can be set using the argument *prompt*. The following prompts are currently available: Box, Boxes, PosNegPoints, PosPoints, HybridA (Boxes + PosPoints), HybridC (Box + PosPoints), and HybridD (Box, PosNegPoints). The number of positive and negative prompts will depend on the argument *area_Thr* (as smaller areas require more points). Currently, HybridA does not support more than one positive point per box.

**2. `main.py` located inside '\_main\_' folder:**

This is the main script where the SAM model is defined and the training process is executed. The 'SAMDataset' function, which is critical for loading and preprocessing the dataset, is included here.

**Key Points:**

-   Handles the setup for training and evaluation of the SAM model.
-   Includes the function 'SAMDataset', which need to be modified for your specific dataset.
-   To handle the prompts, you only need to set the argument *prompt* in the configuration file. If you want to use a different prompt, you need to set it manually.

``` python
  outputs = model(pixel_values=batch["pixel_values"].to(device),
                  input_boxes=input_boxes,# Use this prompt
                  input_points=input_points,# Other prompts set to None
                  input_labels=input_labels, # Other prompts set to None
                  multimask_output=False)  # set to 'True' if you want multi-mask output
```

**Note:** 'input_points' and 'input_labels' prompts should be used together

**3. `dataset.py` located inside 'utils' folder:**

This file provides a template for setting up your dataset, including necessary transformations and preprocessing steps tailored to your data. You don't need to run the 'dataset.py' file directly; the 'SAMDataset' function is invoked within the 'main.py' file during training.

**Key Focus Areas:**

-   'ScaleIntensityRanged' Transform:
    -   For images: Adjust the 'a_min' and 'a_max' parameters to match the intensity range of your dataset.

    -   For labels: Set the 'a_min' and 'a_max' parameters according to the intensity range of your labels. If your labels are already binarized (values of 0 and 1), you can skip this transformation for labels.
-   Handling of Prompts Functions.

**4. `train.py`:**

This file is used to execute the main training script and run the training process.

**5. `inference.py`:**

This file is used to perform testing using the test set.

**6. `requirements.txt`:**

Lists all the Python dependencies required to run this project.

Ensure you install all the required packages using the following command:

``` python
pip install -r requirements.txt
```

<h2>

<u>Quick Start</u>

</h2>

**Install**

-   Step 1:

``` python
pip install -q git+https://github.com/huggingface/transformers.git
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

-   Step 2: Download the SAM weights from [SAM repository](https://github.com/facebookresearch/segment-anything#model-checkpoints)

**Train and Test**

-   Step 1: Modify the config file, select the prompts you wish to use, and adjust the dataset ''SAMDataset' in the 'main.py' file.

-   Step 2: To execute the training, run the 'train.py' file.

-   Step 3: To perform testing, run the 'inference.py' file. Be sure to update the paths to match your setup before executing it.

<h2>

<u>Acknowlegments</u>

</h2>

The code presented in this repository builds upon and integrates concepts from the following resources:

-   [SAM](https://github.com/facebookresearch/segment-anything)

-   [finetune-SAM](https://github.com/mazurowski-lab/finetune-SAM?tab=readme-ov-file)

-   [MedSAM](https://github.com/bowang-lab/MedSAM)

-   [LoRA for SAM](https://github.com/JamesQFreeman/Sam_LoRA)

-   [Medical SAM Adapter](https://github.com/MedicineToken/Medical-SAM-Adapter)

<h2>

<u>Citation</u>

</h2>

Please cite our paper if you use our code or reference our work:

``` python
@article{Siblini2024SAM,
  author    = {Siblini, Lara and Andrade-Miranda, Gustavo and Taguelmimt, Kamilia and Visvkis, Dimitris and Bert, Julien},
  title     = {Optimal Prompting in SAM for Few-Shot and Weakly Supervised Medical Image Segmentation},
  journal = {Proceedings of the 2nd International Workshop on Foundation Models for General Medical AI at MICCAI 2024},
  year      = {2024},
  doi       = {10.1007/978-3-031-73471-7_11},
  url       = {https://link.springer.com/chapter/10.1007/978-3-031-73471-7_11},
  publisher = {Springer},
}

```
