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
file_path = r"D:\Images\T2\ProstateX-0191_t2_tse_tra_Grappa3_2.nii"
img_data, img_affine = read_nifti_file(file_path)
display_image_slices(img_data)  # Display composite image of slices