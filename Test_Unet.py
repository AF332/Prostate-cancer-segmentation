from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization
from keras.optimizers import Adam
from keras.layers import Activation, MaxPooling2D, Concatenate
import tensorflow as tf

def conv_block(input, num_filters):
    convolutional_1 = Conv2D(num_filters, 3, padding = 'same')(input)
    convolutional_1 = BatchNormalization()(convolutional_1)
    convolutional_1 = Activation('relu')(convolutional_1)

    convolutional_2 = Conv2D(num_filters, 3, padding = 'same')(convolutional_1)
    convolutional_2 = BatchNormalization()(convolutional_2)
    convolutional_2 = Activation('relu')(convolutional_2)

    return convolutional_2

def encoder_block(input, num_filters):
    skip_con = conv_block(input, num_filters)
    next_layer = MaxPooling2D(2, 2)(skip_con)

    return skip_con, next_layer

def decoder_block(input, skip_con, num_filters):
    up_samp = Conv2DTranspose(num_filters, (2, 2), strides = 2, padding = 'same')(input)
    sum = Concatenate()([up_samp, skip_con])
    convolution = conv_block(sum, num_filters)

    return convolution

def Unet(input_size = (384, 384, 1)):
    inputs = Input(input_size)

    sk_1, inp_block_2 = encoder_block(inputs, 90)
    sk_2, inp_block_3 = encoder_block(inp_block_2, 180)
    sk_3, inp_block_4 = encoder_block(inp_block_3, 360)
    sk_4, inp_block_5 = encoder_block(inp_block_4, 720)
    sk_5, inp_block_6 = encoder_block(inp_block_5, 1440)

    bridge = conv_block(inp_block_6, 2880)

    decoder_block_5 = decoder_block(bridge, sk_5, 1440)
    decoder_block_4 = decoder_block(decoder_block_5, sk_4, 720)
    decoder_block_3 = decoder_block(decoder_block_4, sk_3, 360)
    decoder_block_2 = decoder_block(decoder_block_3, sk_2, 180)
    decoder_block_1 = decoder_block(decoder_block_2, sk_1, 90)

    outputs = Conv2D(1, 1, padding = 'same', activation = 'sigmoid')(decoder_block_1)

    model = Model(inputs, outputs, name = 'Unet')

    return model

model = Unet(input_size = (384, 384, 1))
model.summary()