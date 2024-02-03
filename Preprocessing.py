import os
import glob
import nibabel as nib
import numpy as np
import tensorflow as tf

# Need to make a environment on one of the computer in G2 for this code.

# Create a nifti data loader
def load_nifti_files(directory_image, directory_mask):  # Defines a function named 'load_nifti_files' whuich takes one argument, 'directory', representing the path to the folder containgthe NIFTI files.
    mri_files_image = sorted(glob.glob(os.path.join(directory_image, "*.nii"))) # The file paths are sorted in a list alphabetically. All files in the directory that match the pattern then constructs a file path patten by joininh the directory path with the filename pattern.
    data_image = [] # Initialising an empty list for the image
    image_shapes = [] # List to store the shapes of each image

    for file in mri_files_image: # For every file in the folder
        image = nib.load(file) # The file is loaded
        image_data = image.get_fdata() # The data from the file is extracted
        num_slices = image_data.shape[-1] # The number of slices are extracted
        image_shapes.append(image_data.shape)
        #print(f"Processing image file: {file}")

        for i in range(num_slices): 
            slice = image_data[:, :, i]
            data_image.append(slice)
            #print(f"Number of slices in {file}: {num_slices}")
            

    mri_files_mask = sorted(glob.glob(os.path.join(directory_mask, "*.nii"))) # Not sure what the last parameter does or how I would need to use it
    data_mask = [] # initialising an empty list for the mask
    mask_shapes = [] # List to store the shapes of each mask
    
    for file in mri_files_mask:
        mask = nib.load(file)
        mask_data = mask.get_fdata()
        num_slices = mask_data.shape[-1]
        mask_shapes.append(mask_data.shape)
        #print(f"Processing mask file: {file}")

        for i in range(num_slices):
            slice = mask_data[:, :, i]
            data_mask.append(slice)
            #print(f"Number of slices in {file}: {num_slices}")

    # Check if all images and masks have the same shape
    if not all(shape == image_shapes[0] for shape in image_shapes):
        print("Warning: Not all images have the same shape.")

    if not all(shape == mask_shapes[0] for shape in mask_shapes):
        print("Warning: Not all masks have the same shape.")

    # Check if each image has a corresponding mask with the same shape
    if len(data_image) == len(data_mask):
        for img, mask in zip(image_shapes, mask_shapes):
            if img != mask:
                print(f"Warning: Image and mask shapes do not match. Image shape: {img}, Mask shape: {mask}")
    else:
        print("Warning: The number of images and masks does not match.")

    return np.array(data_image), np.array(data_mask)
# All the elements present in the list don't have the same shape or length.
# Will have to resize, crop or padd the elements to be consistent but first need to find which one needs this function applied to.

directory_image = r"F:\Images\T2"
directory_mask = r"F:\Masks\T2"
# The shape of the variable do not match up why?
# There's more input images than there is masks

# The good thing is the shape of each slices do match up (384,384) for both the input image and mask for the first image and mask.
# The last image and mask does not match. The last image is (320,320) and the last mask is (384, 384).

# Load Data
data_image, data_mask = load_nifti_files(directory_image, directory_mask)

print("Data Image shape:", data_image.shape)
print("Data Image dtype:", data_image.dtype)
print("Data Mask shape:", data_mask.shape)
print("Data Mask dtype:", data_mask.dtype)