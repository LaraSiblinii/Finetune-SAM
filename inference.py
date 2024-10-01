import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
import seg_metrics.seg_metrics as sg
from _main_.main import *
import csv
import nibabel as nib
from omegaconf import OmegaConf
import wandb
from monai.transforms import (
    AsDiscrete,
    Activations,
    Compose,
)

class configData(object):
    def __init__(self, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])


def DataList_test(datalist,):
    with open(datalist) as f:
        json_data = json.load(f)

    return json_data.keys(),json_data



def RunInference(opt):
    checkpoint_filepath= os.path.join(opt.root_dir,opt.filename)
    model = SamModel.from_pretrained(opt.pretrained_weights)
    if not opt.zeroshot:
        model.load_state_dict(torch.load(checkpoint_filepath)["state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    processor = SamProcessor.from_pretrained(opt.pretrained_weights)
    csv_file= os.path.join(opt.outTest_dir,'metrics.csv')
    
    labels=[1]
    
    model.eval()
    with torch.no_grad():

        PatientList, Data = DataList_test(opt.Testjson_list)
        for patient in PatientList:
            test_dataset= SAMDataset(data_list= Data[patient], processor=processor, opt=opt)
            test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            output_list=[]

            for idx, val_batch in tqdm(enumerate(test_dataloader)):
                val_batch={key: value.to(device) if key != 'label_meta_dict' else value for key, value in val_batch.items()}

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


                outputs = model(pixel_values=val_batch["pixel_values"].to(device),
                                input_points= input_points,
                                input_boxes= input_boxes,
                                input_labels= input_labels,
                                multimask_output=False)
                                
                if opt.combineMask:
                    predicted_val_masks,_ =torch.max(outputs.pred_masks.squeeze(2),dim=1,keepdim=True)
                else:
                    predicted_val_masks = outputs.pred_masks.squeeze(1)

                outputs_decollate = decollate_batch(predicted_val_masks)
                for output in outputs_decollate:
                    output_list.append(opt.post_trans(output)[0])                

            output=torch.stack(output_list,dim=-1)
            file=val_batch["label_meta_dict"]['filename_or_obj'][0].split('/')[-1].split('_')[0] # to modifiy for other datasets
            nii = nib.load(os.path.join(opt.GTLabels_dir, file+'.nii.gz'))
            nib.save(nib.Nifti1Image(output.detach().cpu().numpy(), nii.affine,nii.header), os.path.join(opt.outTest_dir, file+'.nii.gz'))


        metrics=sg.write_metrics(labels=labels,  
                                        pred_path=opt.outTest_dir,
                                        gdth_path=opt.GTLabels_dir,
                                        metrics=['dice','vs','hd95','mdsd'],
                                        csv_file=csv_file
                                )
        
        if  opt.wandb_logger:
            for i in range(len(metrics)):
                opt.wandb_logger.log({f"Test_metrics/{metrics[i]['filename'].split('/')[-1]}-DICE":  metrics[i]['dice'][0]})
                opt.wandb_logger.log({f"Test_metrics/{metrics[i]['filename'].split('/')[-1]}-VS":  metrics[i]['vs'][0]})
                opt.wandb_logger.log({f"Test_metrics/{metrics[i]['filename'].split('/')[-1]}-HD95":  metrics[i]['hd95'][0]})
                opt.wandb_logger.log({f"Test_metrics/{metrics[i]['filename'].split('/')[-1]}-MDSD":  metrics[i]['mdsd'][0]})


if __name__ == "__main__":

    opt={}

    conf_file = sys.argv[1]
    if os.path.exists(conf_file):
        print(f"Config file {conf_file} exist!")
             ###config file ######
        configFile = OmegaConf.load(conf_file)
        opt=configData(configFile)
    opt.combineMask=False
    opt.post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])       


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

    #######create output directory######################################################################
    if not os.path.isdir(opt.outTest_dir):
        os.makedirs(opt.outTest_dir)
    #####################################################################################################

    RunInference(opt)




