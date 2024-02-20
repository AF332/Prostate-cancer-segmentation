import os
import glob
import nibabel as nib
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, SeparableConv2D, Conv2DTranspose, Dropout
from tensorflow.keras.layers import (BatchNormalization, LeakyReLU, Activation, MaxPooling2D, concatenate)
from tensorflow.keras.models import Model

def process_slice(slice): # A function to process a slice to either crop or pad.
    target_size = (384, 384) # The target spatial dimension of all slices.
    pad_width = ((0, 0), (0, 0))  # Initialize padding width for rows and columns. This variable will later be used to determine how much padding is needed on each side of the slice (top-bottom and left-right).
    
    # Calculate padding or cropping needed
    row_diff = target_size[0] - slice.shape[0] # Calculate the difference in height between the target and the image dimensions.
    col_diff = target_size[1] - slice.shape[1] # Calculate the difference in width between the target and the image dimennsions.
    
    if row_diff > 0 or col_diff > 0:  # Check if the slice needs padding as postiive value indicates the slice is smaller
        pad_width = (
            (row_diff // 2, row_diff - row_diff // 2),  # Calculates how much padding is needed on the top of the slice, second argument calculates how much is needed for the bottom, if it is an odd number, the extra pixel is added to the bottom.
            (col_diff // 2, col_diff - col_diff // 2)   # Calculates how much padding is needed on the left of the slice, second argument calculates how much is needed for the right, if it is an odd number, the extra pixel is added to the right.
        )
        processed_slice = np.pad(slice, pad_width, mode='constant', constant_values=0) # Applies padding using numpy to the slice, pad_width tells how much padding is needed and the constant_value specifies that the value 0 is used to pad.
    elif row_diff < 0 or col_diff < 0:  # Checks if the slice is larger than the target dimensions and needs cropping.
        # Calculate start and end indices for cropping
        row_start = -row_diff // 2 # Calculate the starting row index for cropping, '//2' makes the cropping start equally from top and bottom.
        row_end = row_start + target_size[0] # Determines the ending row index for cropping by adding target height to starting row index. Makes sure that the cropped section of the image has the desired height, starting from row_start.
        col_start = -col_diff // 2 # Calculates the starting column index for cropping, equally devides the excess width to be removed from both the left and right sides.
        col_end = col_start + target_size[1] # Determines the ending column index for cropping by adding the target width to col_start. Makes usre the cropped section has the desired width.
        processed_slice = slice[row_start:row_end, col_start:col_end] # Crops the slice to the target dimensions by slicing it from row_start to row_end and from col_start to col_end.
    else:  # No change needed
        processed_slice = slice # Leave slice unchanged if it is already the target size.
    
    return processed_slice # Returns the processed slice.

def load_nifti_files(directory_image, directory_mask):  # Defines a function named 'load_nifti_files' which takes 2 different directory as the arguement, representing the path to the folder containing the images and masks NIFTI files.
    mri_files_image = sorted(glob.glob(os.path.join(directory_image, "*.nii"))) # All files in the directory with the ending .nii are returned as a list of file paths and then sorted lexicographically, which is based on the ASCII values of the characters in the filenames.
    data_image = [] # initialising an empty list for the image

    for file in mri_files_image: # For every file in the folder
        image = nib.load(file) # The file is loaded
        image_data = image.get_fdata() # The data from the file is extracted
        num_slices = image_data.shape[-1] # The number of slices are extracted
        
        for i in range(num_slices): # A loop is created to iterate through each slice in the file.
            slice = process_slice(image_data[:, :, i]) # Accesses the image_data and takes the i-th 2D slice. The first 2 dimensions represents the spatial dimension of the slice and for this case we want it all and process it.
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

directory_image = r"/media/ariffaisal/Crucial X9/Images/T2" # Folder path for the input images
directory_mask = r"/media/ariffaisal/Crucial X9/Masks/T2" # Folder path for the corresponding masks

data_image, data_mask = load_nifti_files(directory_image, directory_mask) # the load_nifti_files function is called with the folders paths provided

def split_dataset(data_image, data_mask, train_ratio = 0.7, validation_ratio = 0.2, test_ratio = 0.1): # A dataset splitting function that takes 2 lists, and splits it to training (0.7), val (0.2), and testing (0.1)
    assert isinstance(data_image, np.ndarray) and isinstance(data_mask, np.ndarray) ## Ensure that data_image and data_mask are NumPy arrays
    paired_data = np.stack((data_image, data_mask), axis=1) ## Stack images and masks together along a new axis to keep each image paired with its corresponding mask
    np.random.shuffle(paired_data) ## Shuffle the paired data along the first axis
    
    # Calculate split indices based on ratios
    total_images = paired_data.shape[0] # Extracts the total number of image-mask pairs. 
    train_end = int(total_images * train_ratio) # Determines how many image-mask pairs will be in the training set.
    validation_end = train_end + int(total_images * validation_ratio) # Determines how many image-mask pairs will be in the validation set.
    
    # Split the data
    train_data = paired_data[:train_end] # Takes all elements form the start of the array up to but not including the train_end variable from the paired_data numpy array.
    validation_data = paired_data[train_end:validation_end] # Takes all elements starting from train_end up to but not including validation_end from the paired_data numpy array.
    test_data = paired_data[validation_end:] # Takes all elements starting from validation_end to the end of the paired_data numpy array.
    
    # Separate the images and masks for each dataset
    train_images, train_masks = train_data[:, 0], train_data[:, 1] # Splits the training images from its masks and puts it into separate variable by calling the different axes.
    validation_images, validation_masks = validation_data[:, 0], validation_data[:, 1] # Splits the val images from its masks into separate variables.
    test_images, test_masks = test_data[:, 0], test_data[:, 1] # Splits the testing images from its masks into separate variables.
    
    return np.array(train_images), np.array(train_masks), np.array(validation_images), np.array(validation_masks), np.array(test_images), np.array(test_masks) # Returns all training, val, and testing variables.

train_images, train_masks, validation_images, validation_masks, test_images, test_masks = split_dataset(data_image, data_mask) # Calls the split_dataset function to be applied to the data_image and data_mask nmupy arrays.

def min_max_normalisation(dataset): # Min, max normalisation function that takes the argument dataset.
    min_max_images = [] # initialising an empty list.
    num_slices = dataset.shape[0] # Extracts the number of slices present in the numpy array.

    for i in range(num_slices): # Iterates through the number of slices present.
        image = dataset[i] # Specifies on one slice.
        #print(i)
        min = np.min(image) # Calculates the minimum value of the slice.
        max = np.max(image) # Calculates the maximum value of the slice.
        min_max = (image - min) / (max - min) # Calculates the the min_max_normalisation pixel values of the slice.
        min_max_images.append(min_max) # Appends it to the list.
    
    return np.array(min_max_images) # Returns the list in a numpy array.

normalised_train_images = min_max_normalisation(train_images) # Applies the min_max_normalisation to the train_images.
#normalised_train_masks = min_max_normalisation(train_masks)
normalised_val_images = min_max_normalisation(validation_images) # Applies the min_max_normalisation to the validation_images.
#normalised_val_masks = min_max_normalisation(validation_masks)
normalised_test_images = min_max_normalisation(test_images) # Applies the min_max_normalisation to the test_images.
#normalised_test_masks = min_max_normalisation(test_masks)

def identify_problematic_slices(data_image): # Problematic slices function that takes in data_image as the argument.
    
    problematic_slices = [] # Creates an empty list to contain all the slices that has a problem.
    for i, slice in enumerate(data_image): # Iterates over data_image, providing both the index (i) of each item and the item itself (slice) at each iteration. 
        min_val = np.min(slice) # Calculate the minimum value of the slice.
        max_val = np.max(slice) # Calculate the maximum value of the slice.
        if min_val == max_val: # Checks if minimum value is equal to the maximum value.
            problematic_slices.append(i) # Appends it to the empty list if it is.
            print(f"Slice {i} might cause division by zero: min and max are {min_val}.") # Prints the index and the problematic slice.
    
    return problematic_slices # Return the list.

problematic_slices_train_images = identify_problematic_slices(train_images) # Call the problematic slices function for the train_images
#problematic_slices_train_masks = identify_problematic_slices(train_masks)
problematic_slices_val_images = identify_problematic_slices(validation_images) # Call the problematic slices function for the validation_images.
#problematic_slices_val_masks = identify_problematic_slices(validation_masks)
problematic_slices_test_images = identify_problematic_slices(test_images) # Call the problematic slices function for the test_images.
#problematic_slices_test_masks = identify_problematic_slices(test_masks)

print("Norm train images shape:", normalised_train_images.shape) # Print the data_image shape to see how many images we have
print("Norm train images dtype:", normalised_train_images.dtype) # Print out data_image item types
print("Train masks shape:", train_masks.shape) # Print the data_mask shape to see how many masks we have
print("Train masks dtype:", train_masks.dtype) # Print out the data_mask item types
print("Norm validation images shape:", normalised_val_images.shape) # Print the data_mask shape to see how many masks we have
print("Norm validation images dtype:", normalised_val_images.dtype) # Print out the data_mask item types
print("Validation masks shape:", validation_masks.shape) # Print the data_mask shape to see how many masks we have
print("Validation masks dtype:", validation_masks.dtype) # Print out the data_mask item types
print("Norm test images shape:", normalised_test_images.shape) # Print the data_mask shape to see how many masks we have
print("Norm test images dtype:", normalised_test_images.dtype) # Print out the data_mask item types
print("Test masks shape:", test_masks.shape) # Print the data_mask shape to see how many masks we have
print("Test masks dtype:", test_masks.dtype) # Print out the data_mask item types

# Create the convolutional blocks
def conv_block(inputs = None, n_filters = 90, batch_norm = False, dropout_prob = 0.4):
    convolutional_1 = SeparableConv2D(n_filters, 2, padding = 'same', kernel_initializer = 'HeNormal')(inputs) # Check the HeNormal parameter what it does
    if batch_norm:
        convolutional_1 = BatchNormalization(axis = 1)(convolutional_1)
    convolutional_1 = LeakyReLU(alpha = 0.2)(convolutional_1)

    convolutional_2 = SeparableConv2D(n_filters, 2, padding = 'same', kernel_initializer = 'HeNormal')(inputs)
    if batch_norm:
        convolutional_2 = BatchNormalization(axis = 1)(convolutional_2)
    convolutional_2 = LeakyReLU(alpha = 0.2)(convolutional_2)

    if dropout_prob > 0:
        convolutional_2 = Dropout(dropout_prob)(convolutional_2)
    
    return convolutional_2

# Create the encoder block
def encoder_block(inputs = None, n_filters = 90, batch_norm = True, dropout_prob = 0.4):
    skip_con = conv_block(inputs, n_filters, batch_norm, dropout_prob)
    next_layer = MaxPooling2D((2,2))(skip_con)

    return skip_con, next_layer

# Create the decoder block
def decoder_block(expansive_input, skip_con, n_filters = 60, batch_norm = False, dropout_prob = 0.4):
    up_samp = Conv2DTranspose(n_filters, 5, strides = 2, padding = 'same', kernel_initializer = 'HeNormal')(expansive_input)
    # Not sure what the kernel initialiser does if it's actually needed
    sum = concatenate([up_samp, skip_con], axis = -1)
    convolution = conv_block(sum, n_filters, batch_norm, dropout_prob)

    return convolution

# Create U-net model
def Unet(input_size = (384, 384, 1), n_filters = 90, n_classes = 2, batch_norm = False, dropouts = [0.4] * 13):
    inputs = Input(input_size)
    
    [sk_1, inp_block_2] = encoder_block(inputs, n_filters, batch_norm, dropout_prob = dropouts[0])
    [sk_2, inp_block_3] = encoder_block(inp_block_2, n_filters * 2, batch_norm, dropout_prob = dropouts[1])
    [sk_3, inp_block_4] = encoder_block(inp_block_3, n_filters * 4, batch_norm, dropout_prob = dropouts[2])
    [sk_4, inp_block_5] = encoder_block(inp_block_4, n_filters * 8, batch_norm, dropout_prob = dropouts[3])
    [sk_5, inp_block_6] = encoder_block(inp_block_5, n_filters * 16, batch_norm, dropout_prob = dropouts[4])
    [sk_6, inp_block_7] = encoder_block(inp_block_6, n_filters * 32, batch_norm, dropout_prob = dropouts[5])

    bridge = conv_block(inp_block_7, n_filters * 64, batch_norm, dropout_prob = dropouts[6])

    decoder_block_6 = decoder_block(bridge, sk_6, n_filters * 32, batch_norm, dropout_prob = dropouts[7])
    decoder_block_5 = decoder_block(decoder_block_6, sk_5, n_filters * 16, batch_norm, dropout_prob = dropouts[8])
    decoder_block_4 = decoder_block(decoder_block_5, sk_4, n_filters * 8, batch_norm, dropout_prob = dropouts[9])
    decoder_block_3 = decoder_block(decoder_block_4, sk_3, n_filters * 4, batch_norm, dropout_prob = dropouts[10])
    decoder_block_2 = decoder_block(decoder_block_3, sk_2, n_filters * 2, batch_norm, dropout_prob = dropouts[11])
    decoder_block_1 = decoder_block(decoder_block_2, sk_1, n_filters, batch_norm, dropout_prob = dropouts[12])

    if n_classes ==2:
        conv10 = SeparableConv2D(1, 1, padding = 'same')(decoder_block_1)
        output = Activation('sigmoid')(conv10)
    else:
        conv10 = SeparableConv2D(1, 1, padding = 'same')(decoder_block_1)
        output = Activation('softmax')(conv10)
    # Not sure if I need the n_classes because the dataset is a binary classification

    model = Model(inputs = inputs, outputs = output, name ='Unet')

    return model

if __name__ == '__main__':
    model = Unet()
    print(model.summary())

# Compile model
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00005),
                loss = 'mse',
                metrics = ['mse', 'mae'])
                # Might need to change the metrics depending on the dataset. USE DICE

history = model.fit(x = normalised_train_images, y = train_masks, batch_size = 32, epochs = 15, verbose = 1) #validation_split = 0.2
# Not sure what the verbose parameter does

# Fid the outputs of the model based on the test inputs
result_images = model.predict(normalised_test_images)

# Find the loss for the test dataset
model.evaluate(normalised_test_images, test_masks)
# Should try to apply a gradient descent function to manually calculate the loss. Might be more accurate. # NOT NEEDED

# Visualise the input, target and out
plt.subplot(2, 3, 1)
plt.imshow(abs(normalised_test_images[17]), cmap = 'gray')
plt.title('Network Input')
plt.imshow(abs(result_images[17]), cmap = 'gray')
plt.title('Network Output')
plt.imshow(abs(test_masks[17]), cmap = 'gray')
plt.title('Test Masks')
plt.imshow(abs(normalised_test_images[10]), cmap = 'gray')
plt.title('Network Input')
plt.imshow(abs(result_images[10]), cmap = 'gray')
plt.title('Network Output')
plt.imshow(abs(test_masks[10]), cmap = 'gray')
plt.title('Test Masks')
plt.tight_layout()

# Visualise the loss convergence
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss Value')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc = 'uperr left')
plt.show()
# For the multi-input U-net model, I need to check if the masks and image are the same dimension using .shape()