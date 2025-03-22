import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Reshape, Multiply, Concatenate, AveragePooling2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Initial Convolutional Layer
    conv_layer = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_layer)
    batch_norm = BatchNormalization()(conv_layer)
    relu_activation = ReLU()(batch_norm)

    # Global Average Pooling
    pooled_output = GlobalAveragePooling2D()(relu_activation)

    # Two Fully Connected Layers
    dense1 = Dense(units=32, activation='relu')(pooled_output)
    dense2 = Dense(units=32, activation='relu')(dense1)

    # Reshape to match the initial feature map dimensions
    reshaped_output = Reshape((1, 1, 32))(dense2)  # Reshaping to (1, 1, 32)

    # Multiply with the initial feature maps to generate weighted feature maps
    weighted_features = Multiply()([relu_activation, reshaped_output])

    # Concatenate with the input layer
    concatenated = Concatenate()([input_layer, weighted_features])

    # Dimensionality reduction and downsampling
    conv_final = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(concatenated)
    pooled_final = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(conv_final)

    # Final Fully Connected Layer for Classification
    flatten_output = Flatten()(pooled_final)
    output_layer = Dense(units=10, activation='softmax')(flatten_output)  # 10 classes for CIFAR-10

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model