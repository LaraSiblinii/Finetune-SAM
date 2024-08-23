**Overview**

The *'dataset.py'* file provides a template for setting up a dataset in your project. It includes transformations and preprocessing steps
tailored to your data. This README explains how to customize the *'dataset.py'* file for different datasets, with a particular focus on the
*'ScaleIntensityRanged'* transform.

**Note:** The SAMDataset function is defined in *'main.py'*. This README focuses on explaining how to modify *'dataset.py'* but does not 
require running this file directly during training.

**Key Transformations:**

**1. 'ScaleIntensityRanged'**

The *'ScaleIntensityRanged'* transform is used to scale the intensity values of your images and labels. This normalization ensures that the input
data is consistent and suitable for model training. Here’s how to configure it based on your dataset:

**For Images:**

-'a_min': This parameter should be set to the minimum intensity value present in your dataset images.

-'a_max': This parameter should be set to the maximum intensity value present in your dataset images.

-'b_min': The minimum value to scale to. Typically set to 0.0 for standardization.

-'b_max': The maximum value to scale to. Typically set to 255.0 for scaling intensity values to the range [0, 255].


**Example Configuration:**

```python
ScaleIntensityRanged(
    keys=['image'],
    a_min=-150,        # min intesisty for IRCAD
    a_max=250,         # max intensity for IRCAD
    b_min=0.0,
    b_max=255.0,
    clip=True
)

ScaleIntensityRanged(
    keys=['image'],
    a_min=0,            # min intesisty for PICAI
    a_max=4743,         # max intensity for PICAI
    b_min=0.0,
    b_max=255.0,
    clip=True
)
```

**For Labels:**

-'a_min': This parameter should be set to the minimum intensity value present in your labels.
-'a_max': This parameter should be set to the maximum intensity value present in your labels.
-'b_min': The minimum value to scale to, usually 0.0.
-'b_max': The maximum value to scale to, often 1.0.

**Important Note:** If your labels are already binarized (i.e., they only contain values 0 and 1), you should skip the ScaleIntensityRanged 
transform for labels.

