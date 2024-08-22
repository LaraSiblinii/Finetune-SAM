import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
import matplotlib.pyplot as plt
from _main_.main import *

if __name__ == "__main__":
    model = SamModel.from_pretrained("facebook/sam-vit-base")
   
    for name, param in model.named_parameters():
        if name.startswith("prompt_encoder") or name.startswith("vision_encoder"):
            print(name)
            param.requires_grad_(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(device)
    optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean') 
    
    root_dir= "/home3/lsiblini/output/"
    filename="model_checkpoints.pt"
    train_output_path= "/home3/lsiblini/output/train_epoch.txt"
    val_output_path= "/home3/lsiblini/output/val_epoch.txt"
    data_dir = "/home3/lsiblini/data/IRCAD"
    images_dir = os.path.join(data_dir, "database_images_nii")
    labels_dir = os.path.join(data_dir, "database_labels_nii")
    json_list = "/home3/lsiblini/data/IRCAD/trainval.json"
    
    tr, val = datafold_read(json_list, data_dir, key="training")

    train_dataset= SAMDataset(data_list= tr, processor=processor)
    train_dataloader = DataLoader(train_dataset, batch_size=10, collate_fn=pad_collate_fn, shuffle=True)

    val_dataset = SAMDataset(data_list=val, processor=processor)
    val_dataloader = DataLoader(val_dataset, batch_size=1, collate_fn=pad_collate_fn, shuffle=True)

    max_epochs = 200
    start_epoch=0


    (
        val_loss_min,
        val_losses_avg,
        train_losses_avg,
        trains_epoch        
    ) = trainer(

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

    plt.savefig("/home3/lsiblini/output/curves_plot.png")
    
