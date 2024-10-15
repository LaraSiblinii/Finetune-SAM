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
#matplotlib.use('Agg')
from torch.optim import Adam
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from transformers import SamModel, SamConfig
#import matplotlib.patches as patches
from transformers import SamProcessor
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import label, find_objects

# Add the parent directory of _main_ to the Python path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from loss_function.loss_functions import seg_loss
from monai.data import decollate_batch
from utils.util import save_images_examples,save_checkpoint,PlotTrVal_curves


from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
    SpatialPadd,
    ScaleIntensityRanged,
    Spacingd,
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

    Returns:
    List of Bounding Box Coordinates: [xmin, ymin, xmax, ymax]
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
    

def get_all_bounding_box(ground_truth_map):
    '''
    This function creates varying bounding box coordinates based on the segmentation contours as prompt for the SAM model
    The padding is random int values between 5 and 20 pixels

    Returns:
    List of Bounding Box Coordinates: [xmin, ymin, xmax, ymax]
    '''
    labeled_mask, num_features = label(ground_truth_map)
    bbox=[]

    if len(np.unique(ground_truth_map)) > 1:

        for regionID in range(num_features):
            y_indices, x_indices = np.where(labeled_mask==regionID+1) 
      
            x_min, x_max = np.min(x_indices), np.max(x_indices) 

            y_min, y_max = np.min(y_indices), np.max(y_indices)

            H, W = ground_truth_map.shape
            x_min = max(0, x_min - np.random.randint(5, 20))
            x_max = min(W, x_max + np.random.randint(5, 20))
            y_min = max(0, y_min - np.random.randint(5, 20))
            y_max = min(H, y_max + np.random.randint(5, 20))

            bbox.append([x_min, y_min, x_max, y_max])

        return bbox
    else:
        return [0, 0, 256, 256]    
       
def get_region_centroids(binary_mask):
    """
    This function identifies distinct regions within a binary mask, places points all over each region,
    and then replaces those points with the midpoint of each region.
    
    Parameters:
    binary_mask (numpy.ndarray): 2D array where 1 represents the regions and 0 represents the background.
    
    Returns:
    all_points: List of lists containing the coordinates of points for each region.
    midpoints: List of midpoints for each region with float coordinates.
    labels: List of labels (1 or 0) indicating the presence of regions.
    """
    
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

def get_region_centroids_boxes_points(binary_mask,size_threshold=0):
    """
    This function identifies distinct regions within a binary mask and calculates their centroids,
    the two top-left, and the two top-right points.
    
    Parameters:
    binary_mask (numpy.ndarray): 2D array where 1 represents the regions and 0 represents the background.
    size_threshold: area of the region.
    
    Returns:
    all_points: List of lists containing the coordinates points for each region.
    midpoints: List of coordinates for the computed midpoints (or centroids) and bounding box corners (top-left and top-right points). The coordinates are based on whether the region is large or small.
    labels: List of labels (1 or 0) indicating the presence of regions.
    """

    labeled_mask, num_features = label(binary_mask)
    regions = find_objects(labeled_mask)

    all_points = []
    midpoints = []
    labels = []
    new_mask = np.zeros_like(binary_mask)
    
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

def get_region_severalpositive_negative_points(binary_mask,size_threshold=20):
    """
    This function identifies distinct regions within a binary mask, calculates their centroids,
    the two top-left, and the two top-right points, and places points outside the regions but between the regions (background points).
    
    Parameters:
    binary_mask (numpy.ndarray): 2D array where 1 represents the regions and -1 represents the background.
    size_threshold: area of the region.
    
    Returns:
    all_points: List of lists containing the coordinates points for each region.
    midpoints: A list of coordinates for the computed midpoints (or centroids) of regions, bounding box corners, and background points.
    labels: List of labels (1 or -1) corresponding to each entry in midpoints. The label 1 is assigned to points derived from actual regions, while -1 is background points.
    """
    
    labeled_mask, num_features = label(binary_mask)
    regions = find_objects(labeled_mask)

    all_points = []
    midpoints = []
    labels = []
    boundary_points=[]
    new_mask = np.zeros_like(binary_mask)

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
    input_boxes = [item['input_boxes'][np.newaxis,:] for item in batch]
    ground_truth_mask = torch.stack([item['ground_truth_mask'] for item in batch])
    
    input_points = [item['input_points'] for item in batch]

    input_labels = [item['input_labels'] for item in batch]

    padded_input_boxes = pad_sequence([boxes.squeeze(0) for boxes in input_boxes], batch_first=True, padding_value=0) 
    padded_input_points = pad_sequence([points.squeeze(0) for points in input_points], batch_first=True, padding_value=0) 
    padded_input_labels = pad_sequence([labels.squeeze(0) for labels in input_labels], batch_first=True, padding_value=-10)
    
    padded_input_boxes = padded_input_boxes
    padded_input_points = padded_input_points.unsqueeze(1)
    padded_input_labels = padded_input_labels.unsqueeze(1)


    return {
        'pixel_values': pixel_values,
        'input_boxes': padded_input_boxes,
        'input_points': padded_input_points,
        'input_labels': padded_input_labels,
        'ground_truth_mask': ground_truth_mask,
    }

    
class SAMDataset(Dataset):
    def __init__(self, data_list, processor,opt):

        self.opt=opt
        self.data_list = data_list
        self.processor = processor
        self.transforms  = Compose([

            LoadImaged(keys=['image', 'label'],image_only=False),

            EnsureChannelFirstd(keys=['image', 'label']),

            #Orientationd(keys=['image', 'label'], axcodes='RA'),
            #Spacingd(keys=['image', 'label'], pixdim=(1, 1), mode=("bilinear", "nearest")),

            ScaleIntensityRanged(keys=['image'], a_min=-150, a_max=250,
                         b_min=0.0, b_max=255.0, clip=True),

            #ScaleIntensityRanged(keys=['label'], a_min=0, a_max=255,
            #             b_min=0.0, b_max=1.0, clip=True),

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

        # convert the grayscale array to RGB (3 channels)/ check when there are 3 modalities as PICAI
        array_rgb = np.dstack((image, image, image)) # numpy_array (256,256,3)

        # convert to PIL image to match the expected input of processor
        image_rgb = Image.fromarray(array_rgb) #PIL.Image.Image

        if self.opt.prompt in ["Boxes", "HybridA", "HybridB","PosPoints","PosNegPoints"]: #do one or multibox
            prompt1 = get_all_bounding_box(ground_truth_mask)#get_bounding_box(ground_truth_mask)#list (123,124,148,152)
        elif self.opt.prompt in ["Box" "HybridC", "HybridD"]:
            prompt1 = get_bounding_box(ground_truth_mask)#get_bounding_box(ground_truth_mask)#list (123,124,148,152)
        else: # no prompts
            prompt1=None


        if self.opt.prompt in ["PosPoints","HybridA", "HybridC",'Boxes','Box']:
            _, prompt2, prompt3 = get_region_centroids_boxes_points(ground_truth_mask,self.opt.area_Thr)
        elif self.opt.prompt in ["PosNegPoints", "HybridB", "HybridD"]:
            _, prompt2, prompt3 = get_region_severalpositive_negative_points(ground_truth_mask,self.opt.area_Thr)
        else: # no prompt
            prompt2,prompt3=None, None

        ##visualize prompts and images
        #prompt2=np.array(prompt2)
        #show_boxes_on_image(image_rgb, prompt1), plt.plot(prompt2[:,0],prompt2[:,1],'ro', markersize=5),plt.show()

        # prepare image and prompt for the model
        if self.opt.prompt !='noPrompts':
            inputs = self.processor(image_rgb,input_boxes=[[prompt1]],input_points=[[prompt2]],input_labels=[[prompt3]],return_tensors="pt")
            # remove batch dimension which the processor adds by default
            inputs = {k: v.squeeze(0) for k, v in inputs.items()} 
        else:
            inputs = self.processor(image_rgb,input_boxes=prompt1,input_points=prompt2,input_labels=prompt3,return_tensors="pt")
            
        # add ground truth segmentation (ground truth image size is 256x256)
        inputs["ground_truth_mask"] = torch.from_numpy(ground_truth_mask.astype(np.int8))
        inputs['label_meta_dict']= data_dict['label_meta_dict']

        return inputs 

def train_epoch(model, train_dataloader, optimizer, epoch, max_epochs, loss, train_output_path,opt):
        model.train()
        start_time = time.time()
        run_loss = AverageMeter()
        with open(train_output_path, "a") as f:
            for i, batch in enumerate(tqdm(train_dataloader)):
                batch= {key: value.to(opt.device) if key != 'label_meta_dict' else value for key, value in batch.items()}

                if opt.prompt=='PosPoints' or opt.prompt=='PosNegPoints':
                    input_points= batch["input_points"].cuda()
                    input_labels= batch["input_labels"].cuda()
                    input_boxes = None
                elif opt.prompt=='Box' or opt.prompt=='Boxes':
                    input_points= None
                    input_labels= None
                    input_boxes = batch["input_boxes"].cuda()
                    opt.combineMask=True
                elif opt.prompt=='HybridA' or opt.prompt=='HybridB':
                    input_points= torch.moveaxis(batch["input_points"],(1,2),(2,1)).cuda()
                    input_labels= torch.moveaxis(batch["input_labels"],1,2).cuda()
                    input_boxes = batch["input_boxes"].cuda()  
                    opt.combineMask=True
                else:# 'HybridC' or 'HybridD'
                    input_points= batch["input_points"].cuda()
                    input_labels= batch["input_labels"].cuda()
                    input_boxes = batch["input_boxes"].cuda()     
                                        

                outputs = model(pixel_values=batch["pixel_values"].to(opt.device),
                            input_points= input_points,
                            input_boxes= input_boxes,
                            input_labels=  input_labels,
                            multimask_output=False)
                if opt.combineMask:
                    predicted_masks,_ =torch.max(outputs.pred_masks.squeeze(2),dim=1,keepdim=True)
                else:
                    predicted_masks = outputs.pred_masks.squeeze(1)

                ground_truth_masks = batch["ground_truth_mask"].float().to(opt.device)
                loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1)) 

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                run_loss.update(loss.item(), n=opt.batch_size)

                ######compute dice training##########################################
                labels_list = decollate_batch(ground_truth_masks[:,np.newaxis,:])
                outputs_list = decollate_batch(predicted_masks)
                output_convert = [opt.post_trans(val_pred_tensor) for val_pred_tensor in outputs_list]

                opt.dice_Train(y_pred=output_convert, y=labels_list)  

                print(
                    "Epoch {}/{} {}/{}".format(epoch, max_epochs, i, len(train_dataloader)),
                    "loss: {:.4f}".format(run_loss.avg),
                    "time {:.2f}s".format(time.time() - start_time),
                    file=f
                )
                start_time = time.time()

            average_diceTraining = opt.dice_Train.aggregate()[0].item()
            print(
                "Epoch {}/{} {}/{}".format(epoch, max_epochs, i, len(train_dataloader)),
                "dice: {:.4f}".format(average_diceTraining),
                file=f
                    )  

        if  opt.wandb_logger:
            opt.wandb_logger.log({"train/loss_epoch":run_loss.avg},step=epoch)
            opt.wandb_logger.log({"train/Dice_epoch":average_diceTraining},step=epoch)

        opt.dice_Train.reset()
        return average_diceTraining,run_loss.avg

def val_epoch(
        model,
        val_dataloader,
        epoch,
        max_epochs,
        val_output_path,
        opt
        ):
        model.eval()
        start_time = time.time()
        run_loss = AverageMeter()
        with torch.no_grad():
            with open(val_output_path, "a") as f:
                for i, val_batch in enumerate(tqdm(val_dataloader)):
                    val_batch={key: value.to(opt.device) if key != 'label_meta_dict' else value for key, value in val_batch.items()}

                    if opt.prompt=='PosPoints' or opt.prompt=='PosNegPoints':
                        input_points= val_batch["input_points"].cuda()
                        input_labels= val_batch["input_labels"].cuda()
                        input_boxes = None
                    elif opt.prompt=='Box' or opt.prompt=='Boxes':
                        input_points= None
                        input_labels= None
                        input_boxes = val_batch["input_boxes"].cuda()
                        opt.combineMask=True
                    elif opt.prompt=='HybridA' or opt.prompt=='HybridB':
                        input_points= torch.moveaxis(val_batch["input_points"],(1,2),(2,1)).cuda()
                        input_labels= torch.moveaxis(val_batch["input_labels"],1,2).cuda()
                        input_boxes = val_batch["input_boxes"].cuda()  
                        opt.combineMask=True
                    else:# 'HybridC' or 'HybridD'
                        input_points= val_batch["input_points"].cuda()
                        input_labels= val_batch["input_labels"].cuda()
                        input_boxes = val_batch["input_boxes"].cuda()    


                    outputs = model(pixel_values=val_batch["pixel_values"].to(opt.device),
                                input_points= input_points,
                                input_boxes= input_boxes,
                                input_labels= input_labels,
                                multimask_output=False)
                                
                    if opt.combineMask:
                        predicted_val_masks,_ =torch.max(outputs.pred_masks.squeeze(2),dim=1,keepdim=True)
                    else:
                        predicted_val_masks = outputs.pred_masks.squeeze(1)


                    ground_truth_masks = val_batch["ground_truth_mask"].float().to(opt.device)
                    val_loss = seg_loss(predicted_val_masks, ground_truth_masks.unsqueeze(1))
                    run_loss.update(val_loss.item(), n=opt.Valbatch_size)


                    ######compute dice training##########################################
                    labels_list = decollate_batch(ground_truth_masks[:,np.newaxis,:])
                    outputs_list = decollate_batch(predicted_val_masks)
                    output_convert = [opt.post_trans(val_pred_tensor) for val_pred_tensor in outputs_list]

                    opt.dice_val(y_pred=output_convert, y=labels_list)  


                    print(
                        "Epoch {}/{} {}/{}".format(epoch, max_epochs, i, len(val_dataloader)),
                        "loss: {:.4f}".format(run_loss.avg),
                        "time {:.2f}s".format(time.time() - start_time),
                        file=f
                    )
                    start_time = time.time()


                average_diceVal = opt.dice_val.aggregate()[0].item()
                print(
                "Epoch {}/{} {}/{}".format(epoch, max_epochs, i, len(val_dataloader)),
                "dice: {:.4f}".format(average_diceVal),
                file=f
                    )  

            if  opt.wandb_logger:
                opt.wandb_logger.log({"val/loss_epoch":run_loss.avg},step=epoch)
                opt.wandb_logger.log({"val/Dice_epoch":average_diceVal},step=epoch)

        opt.dice_val.reset()

                
        return average_diceVal,run_loss.avg
    
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
        opt=None
):
        trains_epoch=[]
        train_losses_avg=[]
        dices_avg_train=[]
        dices_avg_val=[]
        val_losses_avg=[]
        val_dice_min= 0
        for epoch in range(start_epoch, max_epochs):
            print(time.ctime(), "Epoch:", epoch)
            epoch_time = time.time()
            
            train_dice,train_loss = train_epoch(
                model, 
                train_dataloader, 
                optimizer, 
                epoch, 
                max_epochs,
                loss,
                train_output_path,
                opt
             )
            train_avg_loss= np.mean(train_loss)
            trains_epoch.append(int(epoch))
            train_losses_avg.append(train_avg_loss)
            dices_avg_train.append(train_dice)
            
            print(
                "Final training  {}/{}".format(epoch, max_epochs - 1),
                "loss: {:.4f}".format(train_avg_loss),
                "dice: {:.4f}".format(train_dice),
                "time {:.2f}s".format(time.time() - epoch_time),
        )

            val_dice,val_loss = val_epoch(
                model,
                val_dataloader,
                epoch,
                max_epochs,
                val_output_path,opt
            )

            val_avg_loss = np.mean(val_loss)
            val_losses_avg.append(val_avg_loss)
            dices_avg_val.append(val_dice)

                     
            print(
                "Final validation stats {}/{}".format(epoch, max_epochs - 1),
                "loss_Avg: {:.4f}".format(val_avg_loss),
                "dice: {:.4f}".format(val_dice),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
            
            if val_dice_min < val_dice:
                print(f"Model Was Saved! Current Best val dice {val_dice}")
                val_dice_min = val_dice
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    filename,
                    root_dir,
                    best_loss=val_avg_loss,
                    best_dice=val_dice_min
                )
            else:
                print("Model Was Not Saved!")

        print("Training Finished !, Best Dice: ", val_dice_min)

        if opt.saveImg_val:
            save_images_examples(val_dataloader,opt)

        return (
            val_dice_min,
            val_losses_avg,
            train_losses_avg,
            dices_avg_train,
            dices_avg_val,
            trains_epoch
        )
