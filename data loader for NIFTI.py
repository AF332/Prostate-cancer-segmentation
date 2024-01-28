import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

def read_nifti_file(file_path):
    """
    Read a NIfTI file and return the image data as a numpy array.
    
    Args:
    - file_path (str): Path to the NIfTI file.
    
    Returns:
    - img_data (numpy array): The image data.
    - img_affine (numpy array): The affine transformation matrix.
    """
    # Load the NIfTI file
    img = nib.load(file_path)
    
    # Get the image data
    img_data = img.get_fdata()
    
    # Get the affine transformation matrix
    img_affine = img.affine
    
    return img_data, img_affine

def display_image_slices(img_data, num_slices=10):
    """
    Display a composite image of multiple slices from the 3D image data.
    
    Args:
    - img_data (numpy array): The 3D image data.
    - num_slices (int): Number of slices to display.
    """
    # Calculate indices of slices to display
    slice_indices = np.linspace(0, img_data.shape[2] - 1, num_slices, dtype=int)
    
    # Set up the subplot grid
    fig, axes = plt.subplots(1, num_slices, figsize=(15, 3))
    
    # Display each slice
    for i, slice_idx in enumerate(slice_indices):
        img_slice = np.squeeze(img_data[:, :, slice_idx])
        axes[i].imshow(img_slice, cmap='gray')
        axes[i].axis('off')  # Hide axes
        axes[i].set_title(f'Slice {slice_idx}')
    
    plt.tight_layout()
    plt.show()

# Usage:
file_path = r"E:\4th Year\Project\NIFTI\Masks\T2\ProstateX-0003-Finding1-t2_tse_tra_ROI.nii"
img_data, img_affine = read_nifti_file(file_path)
display_image_slices(img_data)  # Display composite image of slices

# The mask for patient 2 has 2 different findings, what should I do about this?
# For the first finding slice 10 and 12 had a segmentation while finding 2 had slice 8 and slice 10 with slice 10 segmentation being different from the 1st finding.
# Mask 3, 5, 12, 21, 23, 28, 31, 33, 35, 37, 38 (has 3), 40 (has 3), 46, 54, 57, 65, 67 (has 3), 68 (has 3), 70, 83, 84, 85 (has 4), 86, 87, 88, 95, 99, 100, 101, 103, 104, 106, 108, 109, 110, 114, 117, 118, 120, 121, 122, 126, 128, 130, 134, 135, 136, 139 (has 3), 140, 141 (has 3), 142 (has 3), 144, 147, 148, 149, 150, 151, 153, 157, 159, 161 (has 3), 163, 170, 171 (has 3), 173, 175, 177 (has 3), 179, 184, 186, 187 (has 3), 189 (has 3), 190, 192, 193 (has 4), 196 (has 3), 199, 200, 202, 203 

import os
import glob
import nibabel as nib
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, SeparableConv2D, Conv2DTranspose, Dropout
from tensorflow.keras.layers import BatchNormalisation, LeakyReLu, Activation, MaxPooling2D, concatenate
from tensorflow.keras.models import Model

# Create a nifti data loader
def load_nifti_files(directory_image, directory_mask):  # Defines a function named 'load_nifti_files' whuich takes one argument, 'directory', representing the path to the folder containgthe NIFTI files.
    mri_files_image = sorted(glob.glob(os.path.join(directory_image, "ProstateX-*.nii"))) # The file paths are sorted in a list alphabetically. All files in the directory that match the pattern then constructs a file path patten by joininh the directory path with the filename pattern.
    data_image = [] # Initialising an empty list for the image

    for file in mri_files_image:
        image = nib.load(file)
        image_data = image.get_fdata()
        num_slices = image_data.shape[-1]

        for i in range(num_slices):
            slice = image_data[:, :, i]
            data_image.append(slice)

    mri_files_mask = sorted(glob.glob(os.path.join(directory_mask, "ProstateX-*.nii")))
    data_mask = [] # initialising an empty list for the mask
    
    for file in mri_files_mask:
        mask = nib.load(file)
        mask_data = mask.get_fdata()
        num_slices = mask_data.shape[-1]

        for i in range(num_slices):
            slice = mask_data[:, :, i]
            data_mask.append(slice)
    
    return np.array(data_image), np.array(data_mask)

# Changing the dataset into tensorflow dataset
def get_dataset(data, batch_size = 8):
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.map(lambda x: tf.expand_dims(x, axis = -1))
    dataset = dataset.map(lambda x: tf.cast(x, tf.float32))
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(len(data))

    return dataset

def min_max_normalisation(dataset.shape[0]):
    min_max_images = [] # initialising an empty list
    for i in range(dataset.shape[0]):
        image = dataset[i]
        min = np.min(image)
        max = np.max(image)
        min_max = (image - min) / (max - min)
        min_max_images.append(min_max)
    
    return np.array(min_max_images)

directory_image = r"E:\4th Year\Project\NIFTI\Images\T2" # Change the path once on external disk
directory_mask = r"E:\4th Year\Project\NIFTI\Masks\T2" # Change the path once on external disk

# Load Data
data_image, data_mask = load_nifti_files(directory_image, directory_mask)

# Normalise the images
normalised_data_image = min_max_normalisation(data_image)
normalised_data_mask = min_max_normalisation(data_mask)

# Create tensorflow datasets
image_dataset = get_dataset(normalised_data_image)
mask_dataset = get_dataset(normalised_data_mask)

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

# Create the convolutional blocks
def conv_block(inputs = None, n_filters = 90, batch_norm = False, dropout_prob = 0.4):
    convolutional_1 = SeparableConv2D(n_filters, 2, padding = 'same', kernel_initializer = 'HeNormal')(inputs)
    if batch_norm:
        convolutional_1 = BatchNormalisation(axis = 1)(convolutional_1)
    convolutional_1 = LeakyReLu(alpha = 0.2)(convolutional_1)

    convolutional_2 = SeparableConv2D(n_filters, 2, padding = 'same', kernel_initializer = 'HeNormal')(inputs)
    if batch_norm:
        convolutional_2 = BatchNormalisation(axis = 1)(convolutional_2)
    convolutional_2 = LeakyReLu(alpha = 0.2)(convolutional_2)

    if dropout > 0:
        convolutional_2 = Dropout(dropout_prob)(convolutional_2)
    
    return convolutional_2

# Create the encoder block
def encoder_block(inputs = None, n_filters = 90, batch_norm = True, dropout_prob = 0.4):
    skip_con = conv_block(inputs, n_filters, batch_norm, dropout_prob)
    next_layer = MaxPooling2D((2,2))(skip_con)

    return skip_con, next_layer

# Create the decoder block
def decoder_block(expansive_input, skip_con, n_filters, batch_norm = False, dropout_prob = 0.4):
    up_samp = Conv2DTranspose(n_filters, 5, strides = 2, padding = 'same', kernel_initializer = 'HeNormal')(expansive_input)
    sum = concatenate([up_samp, skip_con], axis = -1)
    convolution = conv_block(sum, n_filters, batch_norm, dropout_prob)

    return convolution

# Create U-net model
def Unet(input_size = (256, 256, 1), n_filters = 90, n_classes = 2, batch_norm = false, dropout_prob = [0.4] * 13):
    inputs = Input(input_size)

    encoder_block_1 = encoder_block(inputs, n_filters, batch_norm, dropout_prob = dropouts[0])
    encoder_block_2 = encoder_block(encoder_block_1[0], n_filters * 2, batch_norm, dropout_prob = dropouts[1])
    encoder_block_3 = encoder_block(encoder_block_2[0], n_filters * 4, batch_norm, dropout_prob = dropouts[2])
    encoder_block_4 = encoder_block(encoder_block_3[0], n_filters * 8, batch_norm, dropout_prob = dropouts[3])
    encoder_block_5 = encoder_block(encoder_block_4[0], n_filters * 16, batch_norm, dropout_prob = dropouts[4])
    encoder_block_6 = encoder_block(encoder_block_5, n_filters * 32, batch_norm, dropout_prob = dropouts[5])

    bridge = 
# For the multi-input U-net model, I need to check if the masks and image are the same dimension using .shape()