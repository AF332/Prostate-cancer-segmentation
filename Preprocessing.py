import os
import glob
import nibabel as nib
import numpy as np
import tensorflow as tf

# Need to make a environment on one of the computer in G2 for this code.

# Create a nifti data loader
def load_nifti_files(directory_image, directory_mask):  # Defines a function named 'load_nifti_files' which takes 2 different directory as the arguement, representing the path to the folder containing the images and masks NIFTI files.
    mri_files_image = sorted(glob.glob(os.path.join(directory_image, "*.nii"))) # All files in the directory with the ending .nii are returned as a list of file paths and then sorted lexicographically, which is based on the ASCII values of the characters in the filenames.
    image_info = [] # List to store the info of each image
    data_image = [] # List to store all the images  

    for file in mri_files_image: # For every file in the folder
        image = nib.load(file) # The file is loaded
        image_data = image.get_fdata() # The data from the file is extracted
        num_slices = image_data.shape[-1] # The number of slices are extracted
        image_info.append({'filename': file, 'shape': image_data.shape, 'num_slices': num_slices})

        for i in range(num_slices): 
            slice = image_data[:, :, i]
            data_image.append(slice)              

    mri_files_mask = sorted(glob.glob(os.path.join(directory_mask, "*.nii"))) # Not sure what the last parameter does or how I would need to use it
    data_mask = [] # initialising an empty list for the mask
    mask_info = [] # List to store the info of each mask
    
    for file in mri_files_mask:
        mask = nib.load(file)
        mask_data = mask.get_fdata()
        num_slices = mask_data.shape[-1]
        mask_info.append({'filename': file, 'shape': mask_data.shape, 'num_slices': num_slices})
        print(mask_info[0])

        for i in range(num_slices):
            slice = mask_data[:, :, i]
            data_mask.append(slice)

    return np.array(data_image), np.array(data_mask), np.array(image_info), np.array(mask_info)
# All the elements present in the list don't have the same shape or length.
# Will have to resize, crop or padd the elements to be consistent but first need to find which one needs this function applied to.

directory_image = r"F:\Images\T2"
directory_mask = r"F:\Masks\T2"
# The shape of the variable do not match up why?
# There's more input images than there is masks

# The good thing is the shape of each slices do match up (384,384) for both the input image and mask for the first image and mask.
# The last image and mask does not match. The last image is (320,320) and the last mask is (384, 384).

# Load Data
data_image, data_mask, image_info, mask_info = load_nifti_files(directory_image, directory_mask)

print(image_info.shape)
print(mask_info.shape)
# why is the mask when turned into an array doesn't show?

print("Data Image shape:", data_image.shape)
print("Data Image dtype:", data_image.dtype)
print("Data Mask shape:", data_mask.shape)
print("Data Mask dtype:", data_mask.dtype)

# Main problem i can see is that it seems to only show the masks info and not the image info