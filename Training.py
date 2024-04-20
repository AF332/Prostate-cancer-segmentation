from Preprocessing import *
from Test_Unet import *
import tensorflow as tf

mirrored_strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(mirrored_strategy.num_replicas_in_sync))

with mirrored_strategy.scope():
        model = Unet(input_size = (384, 384, 1))
        model.compile(optimizer = Adam(learning_rate = 0.0000005),
                        loss = 'binary_crossentropy',
                        metrics = ["accuracy"])
        
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        
history = model.fit(x = normalised_train_images, y = train_masks, batch_size = 8, epochs = 75, verbose = 1, validation_data = (normalised_val_images, validation_masks)) 

model.save(r"/media/ariffaisal/Crucial X9/Trained models/original_model_120filters_75epoch_addition_1layer")

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss Value')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc = 'upper left')
plt.savefig(r"/media/ariffaisal/Crucial X9/Plots/original_model_120filters_75epoch_addition_1layer_lossconvergence.png")
plt.show()