import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

def read_nifti_file(file_path):
    """
    Read a NIfTI file and return the image data as a numpy array, along with the affine transformation matrix.
    """
    img = nib.load(file_path)  # Load the NIfTI file
    img_data = img.get_fdata()  # Get the image data
    img_affine = img.affine  # Get the affine transformation matrix
    return img_data, img_affine

def display_image_slices(img_data, num_slices=27):
    """
    Display a composite image of multiple slices from the 3D image data in a grid format.
    """
    slice_indices = np.linspace(0, img_data.shape[2] - 1, num_slices, dtype=int)  # Indices of slices to display
    num_rows = int(np.ceil(num_slices / 9))  # Calculate the number of rows needed in the grid

    fig, axes = plt.subplots(num_rows, 9, figsize=(20, num_rows * 2.2))  # Adjust subplot grid and size

    ax = axes.flatten()
    for i, slice_idx in enumerate(slice_indices):
        img_slice = np.squeeze(img_data[:, :, slice_idx])  # Extract slice
        ax[i].imshow(img_slice, cmap='gray')  # Display slice
        ax[i].axis('off')  # Hide axes

    # Hide any unused subplots
    for j in range(i + 1, len(ax)):
        ax[j].axis('off')

    plt.tight_layout()
    plt.show()

# Usage
file_path = r"C:\Users\arifm\Downloads\ProstateX-0000_fid_1_series_7_adc.nii\ProstateX-0000_fid_1_series_7_adc.nii"
img_data, img_affine = read_nifti_file(file_path)
display_image_slices(img_data)  # Display composite image of slices
