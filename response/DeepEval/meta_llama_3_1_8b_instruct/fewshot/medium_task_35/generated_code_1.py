import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, UpSampling2D, Dropout, Dense, Add
from keras.layers import Reshape, Lambda
from keras import backend as K
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Downsampling stage 1
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
    conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)

    # Additional convolution and dropout layers
    conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
    drop1 = Dropout(0.2)(conv3)

    # Upsampling stage 1
    up1 = UpSampling2D(size=(2, 2))(drop1)
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(up1)
    skip1 = Add()([conv4, conv2])
    conv5 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(skip1)

    # Upsampling stage 2
    up2 = UpSampling2D(size=(2, 2))(conv5)
    conv6 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(up2)
    skip2 = Add()([conv6, conv1])
    conv7 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(skip2)

    # Output layer
    output = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')(conv7)
    output = Reshape((10,))(output)

    model = keras.Model(inputs=input_layer, outputs=output)

    return model