import monai
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
    SpatialPadd,
    ScaleIntensityRanged,
    Spacingd,
)

    
class SAMDataset(Dataset):
    def __init__(self, data_list, processor):

        self.data_list = data_list
        self.processor = processor
        self.transforms = transforms = Compose([

            LoadImaged(keys=['image', 'label']),

            EnsureChannelFirstd(keys=['image', 'label']),

            Orientationd(keys=['image', 'label'], axcodes='RA'),

            # resample all training images to a fixed spacing
            Spacingd(keys=['image', 'label'], pixdim=(1.5, 1.5), mode=("bilinear", "nearest")),

            # scale intensities to 0 and 255 to match the expected input intensity range
            ScaleIntensityRanged(keys=['image'], a_min=-150, a_max=250,
                         b_min=0.0, b_max=255.0, clip=True),

            #ScaleIntensityRanged(keys=['label'], a_min=0, a_max=255,
             #            b_min=0.0, b_max=1.0, clip=True),

            SpatialPadd(keys=["image", "label"], spatial_size=(256,256))
            
        ])
   
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_dict = self.data_list[idx]
        image_path = data_dict['image']
        mask_path = data_dict['label']

        # create a dict of images and labels to apply Monai's dictionary transforms
        data_dict = self.transforms({'image': image_path, 'label': mask_path}) #dictionary for a single image; values are paths

        # squeeze extra dimensions and convert to int type for huggingface's models expected inputs
        image = data_dict['image'].squeeze().astype(np.uint8) ##numpy array: (256,256)
        ground_truth_mask = data_dict['label'].squeeze() #metatensor torch.size[(256, 256)]

        # convert the grayscale array to RGB (3 channels)
        array_rgb = np.dstack((image, image, image)) # numpy_array (256,256,3)

        # convert to PIL image to match the expected input of processor
        image_rgb = Image.fromarray(array_rgb) #PIL.Image.Image

        # get bounding box prompt (returns xmin, ymin, xmax, ymax)

        prompt1 = get_bounding_box(ground_truth_mask)#list (123,124,148,152)
        
        _, prompt2, prompt3 = get_region_severalpositive_negative_points(ground_truth_mask)

        # prepare image and prompt for the model
        inputs = self.processor(image_rgb,input_boxes=[[prompt1]],input_points=[[prompt2]],input_labels=[[prompt3]],return_tensors="pt") 

        # remove batch dimension which the processor adds by default
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # add ground truth segmentation (ground truth image size is 256x256)
        inputs["ground_truth_mask"] = torch.from_numpy(ground_truth_mask.astype(np.int8))#add to inputs dictionary a new key 'ground truth'

        return inputs #Dictionary Containing the keys and values for each item, but a list of dictionaries for all data in tr 1280

