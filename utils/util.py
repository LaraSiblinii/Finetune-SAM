from transformers import SamProcessor    
from transformers import SamModel, SamConfig
import torch
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from monai.data import decollate_batch



def save_images_examples(val_dataloader,opt):
    checkpoint_filepath=os.path.join(opt.root_dir,opt.filename)
    config = SamConfig.from_pretrained(opt.pretrained_weights)
    model = SamModel(config)
    model.load_state_dict(torch.load(checkpoint_filepath)["state_dict"])
    model.to(opt.device)

    folder_val=os.path.join(opt.root_dir,'val_results')
    if not os.path.isdir(folder_val):
        os.makedirs(folder_val)
    
    with torch.no_grad():
        for i, val_batch in enumerate(tqdm(val_dataloader)):
            if i%2:
                val_batch={key: value.to(opt.device) for key, value in val_batch.items()}
                                        
                if opt.prompt=='PosPoints' or opt.prompt=='PosNegPoints':
                    input_points= val_batch["input_points"].cuda()
                    input_labels= val_batch["input_labels"].cuda()
                    input_boxes = None
                elif opt.prompt=='Box' or opt.prompt=='Box':
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


                ######compute dice training##########################################
                outputs_list = decollate_batch(predicted_val_masks)
                output_convert = [opt.post_trans(val_pred_tensor) for val_pred_tensor in outputs_list]
                filename=f"Val_{i}.pdf"
                filepath = os.path.join(folder_val, filename)

                if input_points is not None  and input_boxes is not None:
                    if opt.combineMask:
                        points=input_points.squeeze(2).detach().cpu().numpy()[0]
                    else:
                        points=input_points.detach().cpu().numpy()[0][0]
                    boxes=input_boxes.detach().cpu().numpy()[0]
                    show_boxes_on_image(val_batch["pixel_values"].detach().cpu().numpy()[0][0], boxes), plt.plot(points[:,0],points[:,1],'ro', markersize=5)
                    plt.subplot(1,3,2)
                    plt.imshow(ground_truth_masks.detach().cpu().numpy()[0], cmap='copper')
                    plt.axis('off')
                    plt.subplot(1,3,3)
                    plt.imshow(output_convert[0].detach().cpu().numpy()[0], cmap='copper')
                    plt.axis('off')
                    #plt.show()
                    plt.tight_layout()                 
                elif input_points is not None:
                    if opt.combineMask:
                        points=input_points.squeeze(2).detach().cpu().numpy()[0]
                    else:
                        points=input_points.detach().cpu().numpy()[0][0]
                    plt.figure(figsize=(10,10))
                    plt.subplot(1,3,1)
                    plt.imshow(val_batch["pixel_values"].detach().cpu().numpy()[0][0], cmap='gray'), plt.plot(points[:,0],points[:,1],'ro', markersize=5)
                    plt.axis('off')                     
                    plt.subplot(1,3,2)
                    plt.imshow(ground_truth_masks.detach().cpu().numpy()[0], cmap='copper')
                    plt.axis('off')
                    plt.subplot(1,3,3)
                    plt.imshow(output_convert[0].detach().cpu().numpy()[0], cmap='copper')
                    plt.axis('off')
                    #plt.show()
                    plt.tight_layout()                  
                else:
                    boxes=input_boxes.detach().cpu().numpy()[0]
                    show_boxes_on_image(val_batch["pixel_values"].detach().cpu().numpy()[0][0], boxes)
                    plt.axis('off')                     
                    plt.subplot(1,3,2)
                    plt.imshow(ground_truth_masks.detach().cpu().numpy()[0], cmap='copper')
                    plt.axis('off')
                    plt.subplot(1,3,3)
                    plt.imshow(output_convert[0].detach().cpu().numpy()[0], cmap='copper')
                    plt.axis('off')
                    #plt.show()
                    plt.tight_layout()
                
                plt.savefig(filepath)
                plt.close()
            else:
                continue           


def save_checkpoint(model, optimizer,epoch, filename, root_dir, best_loss=100,best_dice=1):
    state_dict = model.state_dict()
    optimizer_state_dict = optimizer.state_dict()
    save_dict = {"epoch": epoch, "best_loss": best_loss,"best_dice": best_dice, "state_dict": state_dict, "optimizer_state_dict": optimizer_state_dict}
    filename = os.path.join(root_dir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)      



def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

def show_boxes_on_image(raw_image, boxes):
    plt.figure(figsize=(10,6))
    plt.subplot(1,3,1)
    plt.imshow(raw_image,cmap='gray')
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('off')                                                    

def PlotTrVal_curves(trains_epoch,train_losses_avg,val_losses_avg,dices_avg_train,dices_avg_val,opt):
    #########plot loss curves##################################
    plt.figure("Training and Validation Loss Curves", (12, 6))

    plt.title("Training and Validation Average Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(trains_epoch, train_losses_avg, label="Train Loss", color="red")
    plt.plot(trains_epoch, val_losses_avg, label="Validation Loss", color="blue")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(opt.root_dir,"Loss_curves.pdf"))


    plt.figure("Training and Validation DICE Curves", (12, 6))
    plt.title("Training and Validation Average DICE")
    plt.xlabel("Epoch")
    plt.ylabel("DICE")
    plt.plot(trains_epoch, dices_avg_train, label="Train DICE", color="red")
    plt.plot(trains_epoch, dices_avg_val, label="Validation DICE", color="blue")
    plt.legend()

    plt.tight_layout()

    plt.savefig(os.path.join(opt.root_dir,"Dice_curves.pdf"))