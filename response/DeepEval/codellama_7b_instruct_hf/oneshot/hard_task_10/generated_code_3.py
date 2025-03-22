import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    # 1. Path with 1x1 convolution
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    # 2. Path with sequence of convolutions: 1x1, followed by 1x7, and then 7x1
    conv2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv3 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    # Concatenate outputs from both paths
    concatenated = Concatenate()([conv1, conv2, conv3])
    # 1x1 convolution to align output dimensions with input image's channel
    conv4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)
    # Branch that merges output of main path and branch through addition
    branch = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch = Concatenate()([conv4, branch])
    # Flatten and fully connected layers for classification
    flattened = Flatten()(branch)
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=10, activation='softmax')(dense1)
    model = keras.Model(inputs=input_layer, outputs=dense2)
    return model