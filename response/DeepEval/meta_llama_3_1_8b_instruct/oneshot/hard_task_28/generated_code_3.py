import keras
from keras.layers import Input, DepthwiseConv2D, BatchNormalization, Conv2D, Add, Lambda, Flatten, Dense
from keras.layers import Reshape, Activation, Multiply, Concatenate, AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 dataset images are 32x32x3
    
    # Main Path
    conv = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(input_layer)
    bn = BatchNormalization()(conv)
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(bn)
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Branch Path
    branch = input_layer

    # Combine Main and Branch Paths
    combined = Add()([conv2, branch])

    # Reshape and Flatten
    reshaped = Reshape(target_shape=(-1,))(combined)
    flattened = Flatten()(reshaped)

    # Classification Layers
    dense1 = Dense(units=64, activation='relu')(flattened)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model