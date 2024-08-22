import os
import sys
import glob
import monai
import torch
import time
import json
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import SimpleITK as sitk
from statistics import mean
from torch.optim import Adam
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from transformers import SamModel, SamConfig
import matplotlib.patches as patches
from transformers import SamProcessor
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import label, find_objects
from monai.metrics import DiceMetric
from torch.nn.functional import threshold
# Add the parent directory of _main_ to the Python path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from loss_functions.loss_functions import seg_loss

from monai.transforms import (
    EnsureChannelFirstd,
    EnsureTyped,
    Compose,
    CropForegroundd,
    CopyItemsd,
    LoadImaged,
    CenterSpatialCropd,
    Invertd,
    OneOf,
    Orientationd,
    MapTransform,
    NormalizeIntensityd,
    RandSpatialCropSamplesd,
    CenterSpatialCropd,
    RandSpatialCropd,
    SpatialPadd,
    ScaleIntensityRanged,
    Spacingd,
    RepeatChanneld,
    ToTensord,
)

"Setup average meter, fold reader, checkpoint saver"

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)

def save_checkpoint(model, epoch, filename, root_dir, best_loss=100):
    state_dict = model.state_dict()
    optimizer_state_dict = optimizer.state_dict()
    save_dict = {"epoch": epoch, "best_loss": best_loss, "state_dict": state_dict, "optimizer_state_dict": optimizer_state_dict}
    filename = os.path.join(root_dir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)

def datafold_read(datalist, basedir, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k in d:
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    train = []
    val = []
    for d in json_data:
        if "fold" in d:
            fold_num = d["fold"]
            if fold_num == 0:
                train.append(d)
            elif fold_num == 1:
                val.append(d)
    return train, val 
    
def get_bounding_box(ground_truth_map):
    '''
    This function creates varying bounding box coordinates based on the segmentation contours as prompt for the SAM model
    The padding is random int values between 5 and 20 pixels
    '''

    if len(np.unique(ground_truth_map)) > 1:

        
        y_indices, x_indices = np.where(ground_truth_map > 0) 
      
        x_min, x_max = np.min(x_indices), np.max(x_indices) 

        y_min, y_max = np.min(y_indices), np.max(y_indices)

        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(5, 20))
        x_max = min(W, x_max + np.random.randint(5, 20))
        y_min = max(0, y_min - np.random.randint(5, 20))
        y_max = min(H, y_max + np.random.randint(5, 20))

        bbox = [x_min, y_min, x_max, y_max]

        return bbox
    else:
        return [0, 0, 256, 256] 
       
def get_region_centroids(binary_mask):

    labeled_mask, num_features = label(binary_mask) #array([0, 1, 2, 3, 4, 5, 6, 7], dtype=int32), 7
    
    regions = find_objects(labeled_mask)

    all_points = []
    midpoints = []
    labels = []
    
    
    for region in regions:
        if region is not None:
            y_indices, x_indices = np.where(labeled_mask[region] > 0)
        
            points = [[x + region[1].start, y + region[0].start] for x, y in zip(x_indices, y_indices)]
            all_points.append(points)
            
            x_coords = np.array([p[0] for p in points])
            y_coords = np.array([p[1] for p in points])
            x_centroid = np.median(x_coords)
            y_centroid = np.median(y_coords)

            midpoints.append([x_centroid, y_centroid])
                     
            labels.append(1)

    if not midpoints:
        default_midpoint = (binary_mask.shape[1] // 2, binary_mask.shape[0] // 2)
        midpoints.append(default_midpoint)
        labels.append(0)

    return all_points, midpoints, labels

def get_region_centroids_boxes_points(binary_mask):

    labeled_mask, num_features = label(binary_mask)
    regions = find_objects(labeled_mask)

    all_points = []
    midpoints = []
    labels = []
    new_mask = np.zeros_like(binary_mask)
    size_threshold = 20
    
    for region in regions:
        if region is not None:
            y_indices, x_indices = np.where(labeled_mask[region] > 0)
            points = [[x + region[1].start, y + region[0].start] for x, y in zip(x_indices, y_indices)]
            all_points.append(points)
            
            if len(points) > size_threshold:

                x_coords = np.array([p[0] for p in points])
                y_coords = np.array([p[1] for p in points])

                x_median = np.median(x_coords)
                y_median = np.median(y_coords)
                
                y_start, y_stop = region[0].start, region[0].stop
                x_start, x_stop = region[1].start, region[1].stop
              
                top_left_point1 = [x_start, y_start]
                top_left_point2 = [x_start, y_stop-1]
                top_right_point1 = [x_stop - 1, y_stop-1]
                top_right_point2 = [x_stop-1, y_start]
                

                for i, top_left in enumerate([top_left_point1, top_left_point2]):
                    if top_left not in points:
                        nearest_point = min(points, key=lambda p: np.linalg.norm(np.array(p) - np.array(top_left)))
                        if nearest_point[0] != top_left[0] or nearest_point[1] != top_left[1]:
                            if i == 0:
                                top_left_point1 = nearest_point
                            else:
                                top_left_point2 = nearest_point

                for i, top_right in enumerate([top_right_point1, top_right_point2]):
                    if top_right not in points:
                        nearest_point = min(points, key=lambda p: np.linalg.norm(np.array(p) - np.array(top_right)))
                        if nearest_point[0] != top_right[0] or nearest_point[1] != top_right[1]: 
                            if i == 0:
                                top_right_point1 = nearest_point
                            else:
                                top_right_point2 = nearest_point      
                
                midpoints.extend([
                    [x_median, y_median],                    
                    [top_left_point1[0], top_left_point1[1]],
                    [top_left_point2[0], top_left_point2[1]],
                    [top_right_point1[0], top_right_point1[1]],
                    [top_right_point2[0], top_right_point2[1]]
                ])

                labels.extend([1, 1, 1, 1, 1])

            else:                
                
                x_coords = np.array([p[0] for p in points])
                y_coords = np.array([p[1] for p in points])


                x_centroid = np.median(x_coords)
                y_centroid = np.median(y_coords)

                
                midpoints.append([x_centroid, y_centroid])

                labels.append(1)

    if not midpoints:
        
        default_midpoint = (binary_mask.shape[1] // 2, binary_mask.shape[0] // 2)
        midpoints.append(default_midpoint)
        labels.append(0)
    
    return all_points, midpoints, labels

def get_region_severalpositive_negative_points(binary_mask):

    labeled_mask, num_features = label(binary_mask)
    regions = find_objects(labeled_mask)

    all_points = []
    midpoints = []
    labels = []
    boundary_points=[]
    new_mask = np.zeros_like(binary_mask)
    size_threshold = 20

    for region in regions:
        if region is not None:
            y_indices, x_indices = np.where(labeled_mask[region] > 0)
            points = [[x + region[1].start, y + region[0].start] for x, y in zip(x_indices, y_indices)]
            all_points.append(points)
            
            if len(points) > size_threshold:

                x_coords = np.array([p[0] for p in points])
                y_coords = np.array([p[1] for p in points])

                x_median = np.median(x_coords)
                y_median = np.median(y_coords)
                
                y_start, y_stop = region[0].start, region[0].stop
                x_start, x_stop = region[1].start, region[1].stop
              
                top_left_point1 = [x_start, y_start]
                top_left_point2 = [x_start, y_stop-1]
                top_right_point1 = [x_stop - 1, y_stop-1]
                top_right_point2 = [x_stop-1, y_start]
                

                for i, top_left in enumerate([top_left_point1, top_left_point2]):
                    if top_left not in points:
                        nearest_point = min(points, key=lambda p: np.linalg.norm(np.array(p) - np.array(top_left)))
                        if nearest_point[0] != top_left[0] or nearest_point[1] != top_left[1]:
                            if i == 0:
                                top_left_point1 = nearest_point
                            else:
                                top_left_point2 = nearest_point

                for i, top_right in enumerate([top_right_point1, top_right_point2]):
                    if top_right not in points:
                        nearest_point = min(points, key=lambda p: np.linalg.norm(np.array(p) - np.array(top_right)))
                        if nearest_point[0] != top_right[0] or nearest_point[1] != top_right[1]: 
                            if i == 0:
                                top_right_point1 = nearest_point
                            else:
                                top_right_point2 = nearest_point      
                
                midpoints.extend([
                    [x_median, y_median],                                     
                    [top_left_point1[0], top_left_point1[1]],
                    [top_left_point2[0], top_left_point2[1]],
                    [top_right_point1[0], top_right_point1[1]],
                    [top_right_point2[0], top_right_point2[1]]
                ])

                labels.extend([1, 1, 1, 1, 1])
            
            else:                
                
                x_coords = np.array([p[0] for p in points])
                y_coords = np.array([p[1] for p in points])


                x_centroid = np.median(x_coords)
                y_centroid = np.median(y_coords)

                
                midpoints.append([x_centroid, y_centroid])
            
                labels.append(1)
    
    if midpoints:
        bounding_boxes = [((region[0].start, region[0].stop), (region[1].start, region[1].stop)) for region in regions]
        boundary_points = []
        for i, bbox1 in enumerate(bounding_boxes[:-1]):
            # Calculate the centroid of bbox1
            mid_x1 = (bbox1[1][0] + bbox1[1][1]) // 2
            mid_y1 = (bbox1[0][0] + bbox1[0][1]) // 2

            min_dist = float('inf')
            min_bbox = None

            for j, bbox2 in enumerate(bounding_boxes):                
                if i == j:
                    continue
                # Calculate the centroid of bbox2
                mid_x2 = (bbox2[1][0] + bbox2[1][1]) // 2
                mid_y2 = (bbox2[0][0] + bbox2[0][1]) // 2

                # Compute the distance between the centroids
                dist = np.sqrt((mid_x2 - mid_x1)**2 + (mid_y2 - mid_y1)**2)
                if dist < min_dist:
                    min_dist = dist
                    min_bbox = bbox2

            if min_bbox is not None:
                # Calculate the midpoint between bbox1 and the closest bbox2
                mid_x = (mid_x1 + (min_bbox[1][0] + min_bbox[1][1]) // 2) // 2
                mid_y = (mid_y1 + (min_bbox[0][0] + min_bbox[0][1]) // 2) // 2

                # Ensure it's outside all regions
                if not any([mid_x, mid_y] in points for points in all_points) and [mid_x, mid_y] not in boundary_points:
                    boundary_points.append([mid_x, mid_y])
                    labels.append(-1)  #Label as background points
        
        midpoints.extend(boundary_points)

    if not midpoints:
        default_midpoint = (binary_mask.shape[1] // 2, binary_mask.shape[0] // 2)
        midpoints.append(default_midpoint)
        labels.append(-1)
    
    return all_points, midpoints, labels

def pad_collate_fn(batch):
    
    pixel_values = torch.stack([item['pixel_values'] for item in batch]) 
    input_boxes = torch.stack([item['input_boxes'] for item in batch])
    ground_truth_mask = torch.stack([item['ground_truth_mask'] for item in batch])
    
    input_points = [item['input_points'] for item in batch]

    input_labels = [item['input_labels'] for item in batch]
    
    padded_input_points = pad_sequence([points.squeeze(0) for points in input_points], batch_first=True, padding_value=0) 
    padded_input_labels = pad_sequence([labels.squeeze(0) for labels in input_labels], batch_first=True, padding_value=-10)
    
    padded_input_points = padded_input_points.unsqueeze(1)
    padded_input_labels = padded_input_labels.unsqueeze(1)
    
    return {
        'pixel_values': pixel_values,
        'input_boxes': input_boxes,
        'input_points': padded_input_points,
        'input_labels': padded_input_labels,
        'ground_truth_mask': ground_truth_mask
    }  
    
class SAMDataset(Dataset):
    def __init__(self, data_list, processor):

        self.data_list = data_list
        self.processor = processor
        self.transforms = transforms = Compose([

            LoadImaged(keys=['image', 'label']),

            EnsureChannelFirstd(keys=['image', 'label']),

            Orientationd(keys=['image', 'label'], axcodes='RA'),

            Spacingd(keys=['image', 'label'], pixdim=(1.5, 1.5), mode=("bilinear", "nearest")),

            ScaleIntensityRanged(keys=['image'], a_min=-150, a_max=250,
                         b_min=0.0, b_max=255.0, clip=True),

            ScaleIntensityRanged(keys=['label'], a_min=0, a_max=255,
                         b_min=0.0, b_max=1.0, clip=True),

            SpatialPadd(keys=["image", "label"], spatial_size=(256,256))
            
        ])
   
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_dict = self.data_list[idx]
        image_path = data_dict['image']
        mask_path = data_dict['label']

        # create a dict of images and labels to apply Monai's dictionary transforms
        data_dict = self.transforms({'image': image_path, 'label': mask_path}) 

        # squeeze extra dimensions and convert to int type for huggingface's models expected inputs
        image = data_dict['image'].squeeze().astype(np.uint8)
        ground_truth_mask = data_dict['label'].squeeze() 

        # convert the grayscale array to RGB (3 channels)
        array_rgb = np.dstack((image, image, image)) # numpy_array (256,256,3)

        # convert to PIL image to match the expected input of processor
        image_rgb = Image.fromarray(array_rgb) #PIL.Image.Image

        prompt1 = get_bounding_box(ground_truth_mask)#list (123,124,148,152)
        
        _, prompt2, prompt3 = get_region_severalpositive_negative_points(ground_truth_mask)

        # prepare image and prompt for the model
        inputs = self.processor(image_rgb,input_boxes=[[prompt1]],input_points=[[prompt2]],input_labels=[[prompt3]],return_tensors="pt") 

        # remove batch dimension which the processor adds by default
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # add ground truth segmentation (ground truth image size is 256x256)
        inputs["ground_truth_mask"] = torch.from_numpy(ground_truth_mask.astype(np.int8))

        return inputs 

def train_epoch(model, train_dataloader, optimizer, epoch, max_epochs, loss, train_output_path):
        model.train()
        start_time = time.time()
        run_loss = AverageMeter()
        with open(train_output_path, "a") as f:
            for i, batch in enumerate(tqdm(train_dataloader)):
                batch= {key: value.to(device) for key, value in batch.items()}              

                outputs = model(pixel_values=batch["pixel_values"].to(device),
                            input_points= batch["input_points"].cuda(),
                            input_boxes= None,
                            input_labels=  batch["input_labels"].cuda(),
                            multimask_output=False)

                predicted_masks = outputs.pred_masks.squeeze(1)
                ground_truth_masks = batch["ground_truth_mask"].float().to(device)
                loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1)) 

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                run_loss.update(loss.item(), n=10)
                print(
                    "Epoch {}/{} {}/{}".format(epoch, max_epochs, i, len(train_dataloader)),
                    "loss: {:.4f}".format(run_loss.avg),
                    "time {:.2f}s".format(time.time() - start_time),
                    file=f
                )
                start_time = time.time()

        return run_loss.avg

def val_epoch(
        model,
        val_dataloader,
        epoch,
        max_epochs,
        val_output_path
        ):
        model.eval()
        start_time = time.time()
        run_loss = AverageMeter()
        with torch.no_grad():
            with open(val_output_path, "a") as f:
                for i, val_batch in enumerate(tqdm(val_dataloader)):
                    val_batch={key: value.to(device) for key, value in val_batch.items()}  

                    outputs = model(pixel_values=val_batch["pixel_values"].to(device),
                                input_points= batch["input_points"].cuda(),
                                input_boxes= None,
                                input_labels= batch["input_labels"].cuda(),
                                multimask_output=False)
                                
                    predicted_val_masks = outputs.pred_masks.squeeze(1)
                    ground_truth_masks = val_batch["ground_truth_mask"].float().to(device)
                    val_loss = seg_loss(predicted_val_masks, ground_truth_masks.unsqueeze(1))
                    run_loss.update(val_loss.item(), n=1)
                    print(
                        "Epoch {}/{} {}/{}".format(epoch, max_epochs, i, len(val_dataloader)),
                        "loss: {:.4f}".format(run_loss.avg),
                        "time {:.2f}s".format(time.time() - start_time),
                        file=f
                    )
                    start_time = time.time()
                
        return run_loss.avg
    
def trainer(
        model, 
        train_dataloader,
        val_dataloader,
        optimizer,
        loss,
        start_epoch,
        max_epochs,
        train_output_path,
        val_output_path,
        filename=None,
        root_dir=None,
):
        trains_epoch=[]
        train_losses_avg=[]
        val_losses_avg=[]
        val_loss_min= 100
        for epoch in range(start_epoch, max_epochs):
            print(time.ctime(), "Epoch:", epoch)
            epoch_time = time.time()
            
            train_loss = train_epoch(
                model, 
                train_dataloader, 
                optimizer, 
                epoch, 
                max_epochs,
                loss,
                train_output_path
            )
            train_avg_loss= np.mean(train_loss)
            trains_epoch.append(int(epoch))
            train_losses_avg.append(train_avg_loss)
            
            print(
                "Final training  {}/{}".format(epoch, max_epochs - 1),
                "loss: {:.4f}".format(train_avg_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
        )

            val_loss = val_epoch(
                model,
                val_dataloader,
                epoch,
                max_epochs,
                val_output_path
            )

            val_avg_loss = np.mean(val_loss)
            val_losses_avg.append(val_avg_loss)
            
            print(
                "Final validation stats {}/{}".format(epoch, max_epochs - 1),
                ", loss_Avg:",
                val_avg_loss,
                ", time {:.2f}s".format(time.time() - epoch_time),
            )
            
            if val_avg_loss < val_loss_min:
                print(f"Model Was Saved! Current Best val loss {val_avg_loss}")
                val_loss_min = val_avg_loss
                save_checkpoint(
                    model,
                    epoch,
                    filename,
                    root_dir,
                    best_loss=val_loss_min,
                )
            else:
                print("Model Was Not Saved!")

        print("Training Finished !, Best loss: ", val_loss_min)
        return (
            val_loss_min,
            val_losses_avg,
            train_losses_avg,
            trains_epoch
        )

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SamModel.from_pretrained("facebook/sam-vit-base") 
optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
 
    
