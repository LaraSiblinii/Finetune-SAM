import monai

seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean') 
