import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

def dl_model():
    # Define the input shape based on the CIFAR-10 dataset
    input_shape = (32, 32, 3)
    num_classes = 10

    # Input layer
    inputs = Input(shape=input_shape)

    # Split the input into three groups along the channel dimension
    split_channels = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(inputs)

    # Apply different convolutional kernels: 1x1, 3x3, and 5x5
    conv_1x1 = Conv2D(32, (1, 1), padding='same', activation='relu')(split_channels[0])
    conv_3x3 = Conv2D(32, (3, 3), padding='same', activation='relu')(split_channels[1])
    conv_5x5 = Conv2D(32, (5, 5), padding='same', activation='relu')(split_channels[2])

    # Concatenate the feature maps from the three groups
    concatenated = Concatenate(axis=-1)([conv_1x1, conv_3x3, conv_5x5])

    # Flatten the concatenated features
    flattened = Flatten()(concatenated)

    # Fully connected layers for classification
    dense_1 = Dense(128, activation='relu')(flattened)
    dense_2 = Dense(num_classes, activation='softmax')(dense_1)

    # Create the model
    model = Model(inputs=inputs, outputs=dense_2)

    return model

# Create the model
model = dl_model()

# Print the model summary
model.summary()