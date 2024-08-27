# Prostate-cancer-segmentation

Overview

This GitHub repository contains the Prostate Cancer Segmentation project, which leverages deep learning to accurately segment prostate cancer from MRI scans. The project utilises a U-Net deep learning model to analyse medical images and corresponding masks stored in the NIfTI format. The repository includes scripts for preprocessing images, training the U-Net model, and evaluating its performance.

## Data Preprocessing

Image and Mask Loading

The preprocessing begins with the loading of MRI images and their corresponding masks using the nibabel library, which is designed to handle NIfTI files. Each NIfTI file may contain multiple slices of images, which are processed individually.

Resizing and Normalisation

Images and masks are processed to fit the model's input requirements:

- Resizing: Each slice is resized to a standard dimension of 384x384 pixels. This involves either padding the images with zeros if they are too small, or cropping them if they are too large, ensuring uniformity across the dataset.
- Normalisation: Image intensities are normalized to a range of 0 to 1 to aid in model training and convergence. This step is crucial for maintaining consistency in pixel value ranges across different images.

## Model Architecture

The U-Net architecture implemented in this project includes:

- Convolutional Blocks: Each block comprises two convolutional layers followed by batch normalisation and ReLU activation. These blocks are used both in the encoding and decoding paths of the network.
- Skip Connections: Essential for U-Net, these connections concatenate feature maps from the encoder to the corresponding decoder stage, which helps the network in localising the segmentation regions.
- Pooling and Transpose Convolution: Pooling reduces the spatial dimensions in the encoder path, while transpose convolutions increase the dimensions in the decoder path, aiding in detailed feature reconstruction for segmentation.

![image](https://github.com/user-attachments/assets/28dad64a-6f68-4474-808c-f4323beadaa8)


## Training

Configuration

The model training is configured with:

- Optimiser: Adam optimiser is used for its efficient computation and low memory requirement. It adjusts the learning rate during training.
- Loss Function: Binary crossentropy is used for the binary classification of pixel-wise segmentation.

Distribution Strategy

Training leverages TensorFlow's tf.distribute.MirroredStrategy to parallelise the training across multiple GPUs. This strategy helps in scaling the training process efficiently, reducing the computational time.

![WhatsApp Image 2024-08-27 at 18 51 55_1466740b](https://github.com/user-attachments/assets/1ad9f48b-46d5-4f2d-84ae-22d09f4b86f1)


## Evaluation and Results

Model Evaluation

The model's effectiveness is assessed using the Intersection over Union (IoU) metric, which measures the overlap between the predicted segmentation and the actual mask. Higher IoU values indicate better model performance.

![image](https://github.com/user-attachments/assets/ddd9038b-094b-4a6e-b767-35c41cf9d6f2)


Visualisation

Results from the model are visualised to compare the predicted segmentation masks against the actual masks. This visualisation helps in qualitatively assessing the model's performance across different test images. the final results of the best performing model of the study is provided below.

![LinkedIn Pic](https://github.com/user-attachments/assets/8bfd0426-4362-4976-aa28-f6d2cfb1adf6)


## Conclusion

The Prostate Cancer Segmentation project demonstrates the applicability of U-Net architectures in medical image analysis. By providing detailed visualisations and metrics, the project not only validates the model's effectiveness but also serves as a foundational framework for further research and application in medical imaging.
