import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
import seg_metrics.seg_metrics as sg
from _main_.main import *
import csv
import nibabel as nib

def crop_center(image, crop_size):
    h, w = image.shape
    start_x = w // 2 - (crop_size // 2)
    start_y = h // 2 - (crop_size // 2)
    return image[start_y:start_y+crop_size, start_x:start_x+crop_size]

# For IRCAD, either for Patient9.json or Patient6.json
def datafold_test_IRCAD(datalist, basedir, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)

    test = []

    for d in json_data[key]:
        image_path = os.path.join(basedir, d["image"])
        label_path = os.path.join(basedir, d["label"])
        test.append({"image": image_path, "label": label_path})

    return test

# For PICAI test.json, were fold_num is the testing patient number
def datafold_test_PICAI(datalist, basedir, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k in d:
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    test= []
    for d in json_data:
        if "fold" in d:
            fold_num = d["fold"]
            if fold_num == 10322: # 20 test cases
                test.append(d)

    return test

if __name__ == "__main__":
    data_dir = "/home3/lsiblini/data/IRCAD"
    json_list = "/home3/lsiblini/data/IRCAD/Patient6.json"
    output_dir= "/home3/lsiblini/output/Testing"
    checkpoint_filepath= "/home3/lsiblini/output/model_checkpoints.pt"
    config = SamConfig.from_pretrained("facebook/sam-vit-base")
    model = SamModel(config)
    model.load_state_dict(torch.load(checkpoint_filepath)["state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    train, val, test = datafold_tst_IRCAD(json_list, data_dir, key="training")
    test_dataset= SAMDataset(data_list= test, processor=processor)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    labels=[0,1]
    dice_scores=[]
    jaccard_scores=[]
    precision_scores=[]
    
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(test_dataloader)):

            outputs = model(pixel_values=batch["pixel_values"].cuda(),
            		input_boxes= None,
            		input_points= batch["input_points"].cuda(),
            		input_labels= batch["input_labels"].cuda(),
                    multimask_output=False)
    
            # compute loss
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().cpu().numpy().squeeze() 
            medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
            medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
            medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
            
            plt.subplot(1,2,1)
            plt.imshow(batch["ground_truth_mask"][0], cmap='copper')
            plt.axis('off')
            plt.subplot(1,2,2)
            plt.imshow(medsam_seg, cmap='copper')
            plt.axis('off')
            plt.tight_layout()
            
            filename=f"test_image_{idx}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath)
            plt.close()
            metrics = sg.write_metrics(labels=[labels[1]],  
                                        gdth_img=ground_truth_masks,
                                        pred_img=medsam_seg,
                                        metrics=['dice','jaccard','precision'])
            
            if isinstance(metrics, list) and metrics:  
                print(type(metrics[0])) 
                if metrics[0]:  
                    keys = metrics[0].keys() 
                    items=metrics[0].items()
                    print("Keys of the first dictionary:", keys) 
                    print("items of first dictionary:", items)
            else:
                print("Metrics is not a non-empty list of dictionaries.")
    
            
            for metric_dict in metrics: 
                dice_value=metric_dict.get('dice') 
                jaccard_value=metric_dict.get('jaccard')
                precision_value=metric_dict.get('precision')
                
                dice_scores.append(dice_value)  
                jaccard_scores.append(jaccard_value)
                precision_scores.append(precision_value)
    
            
            filtered_dice_score = [sublist for sublist in dice_scores if not any(value == 0 for value in sublist)]
            #print("filtered dice score", filtered_dice_score)
            filtered_jaccard_score = [sublist for sublist in jaccard_scores if not any(value == 0 for value in sublist)]
            #print("filtered vs score", filtered_vs_score) 
            filtered_precision_score = [sublist for sublist in precision_scores if not any(value == 0 for value in sublist)]
    
                
        avg_dice= np.mean(filtered_dice_score)
        avg_jaccard= np.mean(filtered_jaccard_score)
        avg_precision= np.mean(filtered_precision_score)
    
        print("Average Dice Score:", avg_dice)
        print("Average Jaccard Score:", avg_jaccard)
        print("Average precision Score:", avg_precision)
    
    
        csv_file="/home3/lsiblini/output/Testing/metrics.csv"
        with open(csv_file, "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Sample", "Dice Score", "Jaccard Score", "precision score"])
            for i, (dice_score, jaccard_score, precision_score) in enumerate(zip(dice_scores, jaccard_scores, precision_scores), start=1):
                writer.writerow([i, dice_score, jaccard_score, precision_score])
    
        # Append average score to the CSV file
        with open(csv_file, "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Average [1] excluding 0's", avg_dice, avg_jaccard, avg_precision])             
          
