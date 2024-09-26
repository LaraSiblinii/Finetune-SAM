import os
import glob
import nibabel as nib
import numpy as np

directory_images = "/home/gustavo/Code/Git_workspace/Finetune-SAM/data/IRCAD2/database_images_nii"
directory_masks = "/home/gustavo/Code/Git_workspace/Finetune-SAM/data/IRCAD2/database_labels_nii"

# Function to extract slices from images and rename them

def extract_and_rename_slices(directory, file_extension):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(file_extension):
                nii = nib.load(os.path.join(root, file))
                data = nii.get_fdata()
                num_slices = data.shape[-1]
                patientName = 'Patient '+ (file.split('-')[0][1] if file.split('-')[0][0]== '0' else file.split('-')[0])
                print(f"Patient Name: {patientName}")
                if not os.path.isdir(os.path.join(root, patientName)):
                    os.makedirs(os.path.join(root, patientName))
                
                for i in range(num_slices):
                    slice_data = data[:, :, i]
                    new_file_name = f"{file.split('.')[0]}_{i+1}.nii.gz"
                    nib.save(nib.Nifti1Image(slice_data, nii.affine,nii.header), os.path.join(root, patientName, new_file_name))
                print(f"Saved patient: {patientName}")

                #os.remove(os.path.join(root, file))
                #print(f"Removed original file: {file}")

# Call the function to extract slices and rename them
        
extract_and_rename_slices(directory_images, 'VE.nii.gz')
extract_and_rename_slices(directory_masks, '.nii.gz')
