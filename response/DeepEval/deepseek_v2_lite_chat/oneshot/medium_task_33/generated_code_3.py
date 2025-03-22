import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Constants
    IMAGE_HEIGHT = 32
    IMAGE_WIDTH = 32
    IMAGE_CHANNELS = 3
    NUM_CLASSES = 10

    # Input layer
    inputs = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))

    # Split into three channel groups
    channel_splits = tf.split(inputs, 3, axis=-1)
    channel_splits = [Lambda(lambda x: x)(split) for split in channel_splits]

    # Define separable convolutions for each channel group
    def separable_conv(filters, size, strides=(1, 1)):
        conv = Conv2D(filters, size, strides=strides, activation='relu', kernel_initializer='he_normal')(channel_splits[0])
        depthwise = Conv2D(filters, (size, size), padding='same', kernel_initializer='he_normal')(conv)
        projected = Conv2D(filters, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(depthwise)
        return projected

    # Apply separable convolutions to each channel group
    conv1 = separable_conv(32, (1, 1))
    conv2 = separable_conv(64, (3, 3))
    conv3 = separable_conv(64, (5, 5))

    # Concatenate the outputs from the three groups
    concat = Concatenate(axis=-1)([conv1, conv2, conv3])

    # Batch normalization and flatten
    bn = BatchNormalization()(concat)
    flat = Flatten()(bn)

    # Fully connected layers
    dense1 = Dense(128, activation='relu')(flat)
    dense2 = Dense(64, activation='relu')(dense1)
    output = Dense(NUM_CLASSES, activation='softmax')(dense2)

    # Model
    model = Model(inputs=inputs, outputs=output)

    return model

# Create the model
model = dl_model()
model.summary()