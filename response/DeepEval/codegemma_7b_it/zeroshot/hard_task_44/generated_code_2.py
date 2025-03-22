import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def residual_block(x, filters):
    # Create a copy of the input for residual connections
    residual = x

    # Perform 3x3 convolution with padding
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.Activation('relu')(x)

    # Perform 1x1 convolution to reduce channels
    x = layers.Conv2D(filters, (1, 1))(x)

    # Add the residual connection
    x = layers.add([x, residual])
    x = layers.Activation('relu')(x)
    return x

def max_pooling_block(x):
    # Perform 3x3 max pooling followed by 1x1 convolution
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (1, 1))(x)
    x = layers.Activation('relu')(x)
    return x

def dl_model():
    # Input layer
    inputs = keras.Input(shape=(32, 32, 3))

    # Block 1: Channel Splitting
    split_input = layers.Lambda(lambda x: tf.split(x, 3, axis=3))(inputs)

    # Extract features from each group
    conv1 = layers.Conv2D(64, (1, 1), padding='same')(split_input[0])
    conv2 = layers.Conv2D(64, (3, 3), padding='same')(split_input[1])
    conv3 = layers.Conv2D(64, (5, 5), padding='same')(split_input[2])

    # Concatenate outputs from each group
    concat = layers.concatenate([conv1, conv2, conv3])

    # Dropout to reduce overfitting
    concat = layers.Dropout(0.2)(concat)

    # Block 2: Feature Fusion
    # Branch 1: 1x1 convolution
    branch1 = layers.Conv2D(64, (1, 1), padding='same')(concat)
    branch1 = layers.Activation('relu')(branch1)

    # Branch 2: 1x1 convolution, then 3x3 convolution
    branch2 = layers.Conv2D(64, (1, 1), padding='same')(concat)
    branch2 = layers.Activation('relu')(branch2)
    branch2 = layers.Conv2D(64, (3, 3), padding='same')(branch2)
    branch2 = layers.Activation('relu')(branch2)

    # Branch 3: 1x1 convolution, then 5x5 convolution
    branch3 = layers.Conv2D(64, (1, 1), padding='same')(concat)
    branch3 = layers.Activation('relu')(branch3)
    branch3 = layers.Conv2D(64, (5, 5), padding='same')(branch3)
    branch3 = layers.Activation('relu')(branch3)

    # Branch 4: 3x3 max pooling, then 1x1 convolution
    branch4 = layers.MaxPooling2D((2, 2))(concat)
    branch4 = layers.Conv2D(64, (1, 1), padding='same')(branch4)
    branch4 = layers.Activation('relu')(branch4)

    # Concatenate outputs from all branches
    concat_branches = layers.concatenate([branch1, branch2, branch3, branch4])

    # Feature fusion
    concat_branches = layers.Conv2D(64, (1, 1), padding='same')(concat_branches)
    concat_branches = layers.Activation('relu')(concat_branches)

    # Flatten and fully connected layer
    flatten = layers.Flatten()(concat_branches)
    outputs = layers.Dense(10, activation='softmax')(flatten)

    # Create and return the model
    model = keras.Model(inputs, outputs)
    return model