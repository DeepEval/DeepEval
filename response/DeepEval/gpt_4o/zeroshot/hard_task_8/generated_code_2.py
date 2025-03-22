import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Reshape, Permute, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    input_shape = (28, 28, 1)  # MNIST dataset image shape
    num_classes = 10  # Number of classes for classification

    # Input layer
    inputs = Input(shape=input_shape)

    # Block 1
    # Primary Path
    x1 = Conv2D(32, (1, 1), padding='same', activation='relu')(inputs)
    x1 = DepthwiseConv2D((3, 3), padding='same', activation='relu')(x1)
    x1 = Conv2D(32, (1, 1), padding='same', activation='relu')(x1)

    # Branch Path
    x2 = DepthwiseConv2D((3, 3), padding='same', activation='relu')(inputs)
    x2 = Conv2D(32, (1, 1), padding='same', activation='relu')(x2)

    # Concatenate features from both paths
    x = Concatenate(axis=-1)([x1, x2])

    # Block 2 - Channel Shuffle
    # Get the shape of the features
    height, width, channels = x.shape[1], x.shape[2], x.shape[3]

    # Reshape and permute dimensions for channel shuffling
    groups = 4
    channels_per_group = channels // groups
    x = Reshape((height, width, groups, channels_per_group))(x)
    x = Permute((1, 2, 4, 3))(x)  # Swap the groups and channels_per_group dimensions
    x = Reshape((height, width, channels))(x)

    # Flatten and add Fully Connected Layer for classification
    x = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # Construct the model
    model = Model(inputs, outputs)

    return model

# Create the model
model = dl_model()

# Print the model summary to understand the architecture
model.summary()