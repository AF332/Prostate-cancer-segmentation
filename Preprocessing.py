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
        image_info.append({'filename': file, 'shape': image_data.shape, 'num_slices': num_slices}) # Appends then filename, the shape of the file, and the number of slices each file contains to the image_info variable.

        for i in range(num_slices): # A loop is created to iterate through each slice in the file.
            slice = image_data[:, :, i] # Accesses the image_data and takes the i-th 2D slice. The first 2 dimensions represents the spatial dimension of the slice and for this case with want it all.
            data_image.append(slice) # The extracted slice is then appended to the data_image list.              

    mri_files_mask = sorted(glob.glob(os.path.join(directory_mask, "*.nii"))) # Not sure what the last parameter does or how I would need to use it
    data_mask = [] # initialising an empty list for the mask
    mask_info = [] # List to store the info of each mask
    
    for file in mri_files_mask: # For every file in the folder
        mask = nib.load(file) # The file is loaded
        mask_data = mask.get_fdata() # The data from the file is extracted
        num_slices = mask_data.shape[-1] # The number of slices are extracted
        mask_info.append({'filename': file, 'shape': mask_data.shape, 'num_slices': num_slices}) # Appends then filename, the shape of the file, and the number of slices each file contains to the mask_info variable.

        print(mask_info[0]) # Just to check whether they are any items in the list (there is)

        for i in range(num_slices): # A loop is created to iterate through each slice in the file
            slice = mask_data[:, :, i] # Accesses the mask_data and takes the i-th 2D slice. The first 2 dimensions represent the spatial dimension of the which we want all of it.
            data_mask.append(slice) # he extracted slice is then appended to the data_mask list.

    return np.array(data_image), np.array(data_mask), np.array(image_info), np.array(mask_info) # The array forms of the data_image, data_mask, image_info, and mask_info are returned.

"""All the elements present in the list don't have the same shape or length.
Will have to resize, crop or padd the elements to be consistent but first need to find which one needs this function applied to."""

directory_image = r"F:\Images\T2" # Folder path for the input images
directory_mask = r"F:\Masks\T2" # Folder path for the corresponding masks

"""The shape of the variable do not match up why? There's more input images than there is masks (fixed this by changing the pattern structure of the mask)"""

"""The last image and mask does not match. The last image is (320,320) and the last mask is (384, 384)."""

data_image, data_mask, image_info, mask_info = load_nifti_files(directory_image, directory_mask) # the load_nifti_files function is called with the folders paths provided

print(image_info.shape) # Print the shape of the image_info variable to check if it's correct
print(mask_info.shape) # Print the mask_info variable to check if its's correct (it's empty after converting into a numpy array)

"""why is the mask when turned into an array doesn't show?"""

print("Data Image shape:", data_image.shape) # Print the data_image shape to see how many images we have
print("Data Image dtype:", data_image.dtype) # Print out data_image item types
print("Data Mask shape:", data_mask.shape) # Print the data_mask shape to see how many masks we have
print("Data Mask dtype:", data_mask.dtype) # Print out the data_mask item types

"""Main problem i can see is that it seems to only show the masks info and not the image info (maybe the terminal window is too small to fit everything in but when
I print the shapes they match and have a lot more images and masks then when I printed the variable)"""