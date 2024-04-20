from Preprocessing import *
from tensorflow import keras
import tensorflow as tf

model = keras.models.load_model(r"/media/ariffaisal/Crucial X9/Trained models/original_model_120filters_75epoch_addition_1layer")
model.summary()

# Fid the outputs of the model based on the test inputs
result_images = model.predict(normalised_test_images)
binary_result_images = tf.where(result_images > 0.0001, 1, 0) 
binary_result_images_squeezed = np.squeeze(binary_result_images)

# Find the loss for the test dataset
test_loss, test_accuracy= model.evaluate(normalised_test_images, test_masks)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# IOU
intersection = np.logical_and(test_masks, binary_result_images_squeezed)
union = np.logical_or(test_masks, binary_result_images_squeezed)
iou_score = np.sum(intersection) / np.sum(union)
print("IoU score is: ", iou_score)

# List of image indices you want to plot
image_indices = [17, 10, 100, 150, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23,
                 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
                 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57]

# Number of images to plot
num_images = len(image_indices)

# Setup figure and axes
fig, axs = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))  # Adjust the figure size as needed

for i, img_idx in enumerate(image_indices):
    # Plot Network Input
    axs[i, 0].imshow(abs(normalised_test_images[img_idx]), cmap='gray')
    axs[i, 0].title.set_text('Network Input')
    axs[i, 0].axis('off') 

    # Plot Network Output
    axs[i, 1].imshow(abs(binary_result_images[img_idx]), cmap='gray')
    axs[i, 1].title.set_text('Network Output')
    axs[i, 1].axis('off')

    # Plot Test Masks
    axs[i, 2].imshow(abs(test_masks[img_idx]), cmap='gray')
    axs[i, 2].title.set_text('Test Masks')
    axs[i, 2].axis('off')

plt.tight_layout()
plt.savefig(r"/media/ariffaisal/Crucial X9/Plots2/original_model_120filters_75epoch_addition_1layer_thres0.0001.png")
plt.show()