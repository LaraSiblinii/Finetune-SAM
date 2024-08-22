''' 
if you want to resume training from a saved checkpoint, you can use this file. 
For example, if you have already trained the model for 100 epochs and wish to 
continue from that point, this file will allow you to do so.
'''

import sys
from pathlib import Path
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '_main_')))
import main

if __name__ == "__main__":

    config = SamConfig.from_pretrained("facebook/sam-vit-base")
    model = SamModel(config)
    checkpoint_path = "/home3/lsiblini/output/Results_SAM_noMasks/VEELA/train_100/model_100.pt" #include the path to the checkpoints
    checkpoint=torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model.to(device)
    optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")    
    
    def trainer_checkpoints(
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
        val_loss_min= checkpoint['best_loss'] 
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
                train_output_path,
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
                val_output_path,
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
    
    
    root_dir= "/home3/lsiblini/output/Results_SAM_noMasks/VEELA/train_200/" #new path to save tnew model checkpoints
    filename="model_200.pt"
    data_dir = "/home3/lsiblini/data/IRCAD"
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "portal_labels")
    json_list = "/home3/lsiblini/data/IRCAD/trainval.json"
    train_output_path= "/home3/lsiblini/output/train_200/train-epoch-200"
    val_output_path= "/home3/lsiblini/output/train_200/val-epoch-200"     
    
    tr, val = datafold_read(json_list, data_dir, key="training")

    train_dataset= SAMDataset(data_list= tr, processor=processor)
    train_dataloader = DataLoader(train_dataset, batch_size=10, collate_fn=pad_collate_fn, shuffle=True)

    val_dataset = SAMDataset(data_list=val, processor=processor)
    val_dataloader = DataLoader(val_dataset, batch_size=1, collate_fn=pad_collate_fn, shuffle=True)

    # Freeze the encoder weights for finetuning (only update gradients for mask decoder)
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            print(name)
            param.requires_grad_(False)

    max_epochs = 200
    start_epoch= checkpoint['epoch'] #or directly specify the number of epoch were you wish to start to avoid repeated epochs 

    #Excute training

    (
        val_loss_min,
        val_losses_avg,
        train_losses_avg,
        trains_epoch
    ) = trainer_checkpoints(

        model=model, 
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        loss=seg_loss,
        start_epoch=start_epoch,
        max_epochs=max_epochs,
        train_output_path=train_output_path,
        val_output_path=val_output_path,
        filename=filename,
        root_dir=root_dir,
    )
    print(f"train completed, best average loss: {val_loss_min:.4f} ")

    plt.figure("Training and Validation Curves", (12, 6))

    plt.title("Training and Validation Average Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(trains_epoch, train_losses_avg, label="Train Loss", color="red")
    plt.plot(trains_epoch, val_losses_avg, label="Validation Loss", color="blue")
    plt.legend()

    plt.tight_layout()

    plt.savefig("/home3/lsiblini/output/train_200/curves_plot_200.png")
