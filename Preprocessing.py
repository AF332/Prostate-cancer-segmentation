import os
import glob
import nibabel as nib
import numpy as np
import tensorflow as tf

"""Need to make a environment on one of the computer in G2 for this code."""

def process_slice(slice):
    target_size = (384, 384)
    pad_width = ((0, 0), (0, 0))  # Initialize padding width for rows and columns
    
    # Calculate padding or cropping needed
    row_diff = target_size[0] - slice.shape[0]
    col_diff = target_size[1] - slice.shape[1]
    
    if row_diff > 0 or col_diff > 0:  # Padding
        pad_width = (
            (row_diff // 2, row_diff - row_diff // 2),  # Pad rows
            (col_diff // 2, col_diff - col_diff // 2)   # Pad columns
        )
        processed_slice = np.pad(slice, pad_width, mode='constant', constant_values=0)
    elif row_diff < 0 or col_diff < 0:  # Cropping
        # Calculate start and end indices for cropping
        row_start = -row_diff // 2
        row_end = row_start + target_size[0]
        col_start = -col_diff // 2
        col_end = col_start + target_size[1]
        processed_slice = slice[row_start:row_end, col_start:col_end]
    else:  # No change needed
        processed_slice = slice
    
    return processed_slice

def load_nifti_files(directory_image, directory_mask):  # Defines a function named 'load_nifti_files' which takes 2 different directory as the arguement, representing the path to the folder containing the images and masks NIFTI files.
    mri_files_image = sorted(glob.glob(os.path.join(directory_image, "*.nii"))) # All files in the directory with the ending .nii are returned as a list of file paths and then sorted lexicographically, which is based on the ASCII values of the characters in the filenames.
    data_image = [] # initialising an empty list for the image

    for file in mri_files_image: # For every file in the folder
        image = nib.load(file) # The file is loaded
        image_data = image.get_fdata() # The data from the file is extracted
        num_slices = image_data.shape[-1] # The number of slices are extracted
        
        for i in range(num_slices): # A loop is created to iterate through each slice in the file.
            slice = process_slice(image_data[:, :, i]) # Accesses the image_data and takes the i-th 2D slice. The first 2 dimensions represents the spatial dimension of the slice and for this case with want it all.
            data_image.append(slice) # The extracted slice is then appended to the data_image list.              

    mri_files_mask = sorted(glob.glob(os.path.join(directory_mask, "*.nii"))) # Not sure what the last parameter does or how I would need to use it
    data_mask = [] # initialising an empty list for the mask

    for file in mri_files_mask: # For every file in the folder
        mask = nib.load(file) # The file is loaded
        mask_data = mask.get_fdata() # The data from the file is extracted
        num_slices = mask_data.shape[-1] # The number of slices are extracted
        
        for i in range(num_slices): # A loop is created to iterate through each slice in the file
            slice_2 = process_slice(mask_data[:, :, i]) # Accesses the mask_data and takes the i-th 2D slice. The first 2 dimensions represent the spatial dimension of the which we want all of it.
            data_mask.append(slice_2) # he extracted slice is then appended to the data_mask list.

    return np.array(data_image), np.array(data_mask) # The array forms of the data_image and data_mask are returned.


"""Should I split my dataset into training, validation, and test first then normalise or normalise first? Don't think the test needs validating"""

def min_max_normalisation(dataset.shape[0]):
    min_max_images = [] # initialising an empty list
    for i in range(dataset.shape[0]):
        image = dataset[i]
        min = np.min(image)
        max = np.max(image)
        min_max = (image - min) / (max - min)
        min_max_images.append(min_max)
    
    return np.array(min_max_images)

directory_image = r"F:\Images\T2" # Folder path for the input images
directory_mask = r"F:\Masks\T2" # Folder path for the corresponding masks

data_image, data_mask = load_nifti_files(directory_image, directory_mask) # the load_nifti_files function is called with the folders paths provided

print("Data Image shape:", data_image.shape) # Print the data_image shape to see how many images we have
print("Data Image dtype:", data_image.dtype) # Print out data_image item types
print("Data Mask shape:", data_mask.shape) # Print the data_mask shape to see how many masks we have
print("Data Mask dtype:", data_mask.dtype) # Print out the data_mask item types

normalised_data_image = min_max_normalisation(data_image)
normalised_data_mask = min_max_normalisation(data_mask)

# Split the datasets into training, validation, testing
total_images = len(normalised_data_image)
train_size = int(0.7 * total_images)
val_size = int(0.2 * total_images)
test_size = int(0.1 * total_images)

train_images = image_dataset.take(train_size) # train images
remaining_images = image_dataset.skip(train_size)
val_images = remaining_images.take(val_size) # Validation images
test_images = remaining_images.skip(val_size) # Test images

train_masks = mask_dataset.take(train_size) # Train mask images
remaining_masks = mask_dataset.skip(train_size)
val_masks = remaining_masks.take(val_size) # Validation mask images
test_masks = remaining_masks.skip(val_size) # Test mask images