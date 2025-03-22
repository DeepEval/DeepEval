import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_layer = keras.Input(shape=(28, 28, 1))  

    # Block 1: Multi-Scale Max Pooling
    branch1 = layers.MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    branch2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    branch3 = layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)

    branch1 = layers.Flatten()(branch1)
    branch1 = layers.Dropout(0.25)(branch1)

    branch2 = layers.Flatten()(branch2)
    branch2 = layers.Dropout(0.25)(branch2)

    branch3 = layers.Flatten()(branch3)
    branch3 = layers.Dropout(0.25)(branch3)

    merged_block1 = layers.Concatenate()([branch1, branch2, branch3])

    # Fully Connected and Reshaping
    dense_layer = layers.Dense(128, activation='relu')(merged_block1)
    reshaped_layer = layers.Reshape((1, 128))(dense_layer)

    # Block 2: Multi-Scale Convolutional Feature Extraction
    branch4_conv1 = layers.Conv2D(32, (1, 1))(reshaped_layer)
    branch4_pool = layers.AveragePooling2D(pool_size=(2, 2))(branch4_conv1)
    branch4_conv2 = layers.Conv2D(64, (1, 1))(branch4_pool)

    branch5_conv1 = layers.Conv2D(32, (1, 1))(reshaped_layer)
    branch5_conv2 = layers.Conv2D(64, (1, 7))(branch5_conv1)
    branch5_conv3 = layers.Conv2D(64, (7, 1))(branch5_conv2)

    branch6_conv1 = layers.Conv2D(32, (1, 1))(reshaped_layer)
    branch6_conv2 = layers.Conv2D(64, (1, 7))(branch6_conv1)
    branch6_conv3 = layers.Conv2D(64, (7, 1))(branch6_conv2)

    branch7_conv1 = layers.Conv2D(32, (1, 1))(reshaped_layer)

    merged_block2 = layers.Concatenate(axis=-1)([branch4_conv2, branch5_conv3, branch6_conv3, branch7_conv1])

    # Classification Layers
    flatten = layers.Flatten()(merged_block2)
    dense_layer1 = layers.Dense(128, activation='relu')(flatten)
    output_layer = layers.Dense(10, activation='softmax')(dense_layer1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model