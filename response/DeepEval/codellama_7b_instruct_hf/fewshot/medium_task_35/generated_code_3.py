import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, UpSampling2D, Concatenate, Add
from keras.models import Model
from keras.applications.vgg16 import VGG16

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first stage of convolution and max pooling
    conv_block_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_shape)
    maxpool_block_1 = MaxPooling2D((2, 2))(conv_block_1)

    # Define the second stage of convolution and max pooling
    conv_block_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(maxpool_block_1)
    maxpool_block_2 = MaxPooling2D((2, 2))(conv_block_2)

    # Define the third stage of convolution and max pooling
    conv_block_3 = Conv2D(128, (3, 3), activation='relu', padding='same')(maxpool_block_2)
    maxpool_block_3 = MaxPooling2D((2, 2))(conv_block_3)

    # Define the skip connections
    upsample_block_1 = UpSampling2D((2, 2))(maxpool_block_2)
    skip_connection_1 = Concatenate()([upsample_block_1, conv_block_2])

    upsample_block_2 = UpSampling2D((2, 2))(maxpool_block_1)
    skip_connection_2 = Concatenate()([upsample_block_2, conv_block_1])

    # Define the final convolutional layers
    conv_block_final = Conv2D(128, (3, 3), activation='relu', padding='same')(skip_connection_1)
    conv_block_final = Conv2D(64, (3, 3), activation='relu', padding='same')(conv_block_final)
    conv_block_final = Conv2D(32, (3, 3), activation='relu', padding='same')(conv_block_final)

    # Define the output layer
    output_layer = Conv2D(10, (1, 1), activation='softmax')(conv_block_final)

    # Create the model
    model = Model(inputs=input_shape, outputs=output_layer)

    return model