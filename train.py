import sys
from pathlib import Path
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import random
from monai.utils import set_determinism
import wandb
from monai.metrics import DiceMetric
from inference import RunInference
#os.environ["WANDB_MODE"]="offline"
#matplotlib.use('Agg')

from monai.transforms import (
    AsDiscrete,
    Activations,
    Compose
)



parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from _main_.main import *


class configData(object):
    def __init__(self, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])



if __name__ == "__main__":

    opt={}

    conf_file = sys.argv[1]
    if os.path.exists(conf_file):
        print(f"Config file {conf_file} exist!")
             ###config file ######
        configFile = OmegaConf.load(conf_file)
        opt=configData(configFile)
    opt.combineMask=False
 ####set deterministic training #######
    set_determinism(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
##########################################


###########ask for wandb #####################
    if opt.wandb_act:
        if not os.path.isdir(opt.dir_wandb):
            os.makedirs(opt.dir_wandb)
        opt.wandb_logger = wandb.init(project=opt.project_name,
                                      entity=opt.entity,
                                      config=opt,
                                      name=opt.nameRun,
                                    dir=opt.dir_wandb)
    else:                                
        opt.wandb_logger=None
###############################################################


###### metrics to evaluate dice performance during training###############################
    opt.dice_val = DiceMetric(include_background=True, reduction="mean", get_not_nans=True)
    opt.dice_Train = DiceMetric(include_background=True, reduction="mean", get_not_nans=True)
    opt.post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])       
#############################################################################################


#######create output directory######################################################################
    if not os.path.isdir(opt.root_dir):
        os.makedirs(opt.root_dir)
#####################################################################################################

### Load the model weights
    model = SamModel.from_pretrained(opt.pretrained_weights)


   # Freeze the encoder weights for finetuning (only update gradients for mask decoder)
    for name, param in model.named_parameters():
        if opt.prompt != 'noPrompts':
            if name.startswith("prompt_encoder") or name.startswith("vision_encoder"): # finetune only decoder
                param.requires_grad_(False)
        else:
            if name.startswith("vision_encoder"): # finetune prompt and decoder
                param.requires_grad_(False)

    opt.device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(opt.device)
    print(opt.device)

    if opt.wandb_logger:
        opt.wandb_logger.watch(model,log="all")

    # optimizer and sam processor to prepare the data
    optimizer = Adam(model.mask_decoder.parameters(), lr=opt.lr, weight_decay=0)
    processor = SamProcessor.from_pretrained(opt.pretrained_weights)
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean') 
    
    tr, val = datafold_read(opt.json_list, opt.data_dir, key="training")

    train_dataset= SAMDataset(data_list= tr, processor=processor,opt=opt)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, collate_fn=pad_collate_fn, shuffle=True)

    val_dataset = SAMDataset(data_list=val, processor=processor,opt=opt)
    val_dataloader = DataLoader(val_dataset, batch_size=opt.Valbatch_size, collate_fn=pad_collate_fn, shuffle=True)

    max_epochs = opt.max_epochs
    start_epoch=0

######## training ##############
    (
        val_dice_min,
        val_losses_avg,
        train_losses_avg,
        dices_avg_train,
        dices_avg_val,
        trains_epoch        
    ) = trainer(
        model=model, 
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        loss=seg_loss,
        start_epoch=start_epoch,
        max_epochs=max_epochs,
        train_output_path=opt.train_output_path,
        val_output_path=opt.val_output_path,
        filename=opt.filename,
        root_dir=opt.root_dir,
        opt=opt
    )

    print(f"train completed, best average loss: {val_dice_min:.4f} ")

    #########plot loss curves##################################
    PlotTrVal_curves(trains_epoch,train_losses_avg,val_losses_avg,dices_avg_train,dices_avg_val,opt)


    if opt.RunTest:
    #######create output directory######################################################################
        if not os.path.isdir(opt.outTest_dir):
            os.makedirs(opt.outTest_dir)
    #####################################################################################################
        RunInference(opt)


  
