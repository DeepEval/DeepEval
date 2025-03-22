import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model
from keras.applications import VGG16

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Downsampling stage 1
    conv1 = Conv2D(32, (3, 3), strides=(2, 2), activation='relu')(input_layer)
    maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv1)

    # Downsampling stage 2
    conv2 = Conv2D(64, (3, 3), strides=(2, 2), activation='relu')(maxpool1)
    maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv2)

    # Feature extraction stage
    conv3 = Conv2D(128, (3, 3), strides=(2, 2), activation='relu')(maxpool2)
    conv4 = Conv2D(128, (3, 3), strides=(2, 2), activation='relu')(conv3)
    dropout = Dropout(0.2)(conv4)

    # Upsampling stage 1
    conv5 = Conv2D(64, (3, 3), strides=(2, 2), activation='relu')(dropout)
    upconv1 = Conv2D(64, (3, 3), strides=(2, 2), activation='relu')(conv5)

    # Upsampling stage 2
    conv6 = Conv2D(32, (3, 3), strides=(2, 2), activation='relu')(upconv1)
    upconv2 = Conv2D(32, (3, 3), strides=(2, 2), activation='relu')(conv6)

    # Output layer
    conv7 = Conv2D(10, (1, 1), activation='softmax')(upconv2)

    model = Model(inputs=input_layer, outputs=conv7)

    return model