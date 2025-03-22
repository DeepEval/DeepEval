import tensorflow as tf
from tensorflow.keras import datasets, models
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Lambda, MaxPooling2D, Concatenate
from tensorflow.keras.models import Model
import numpy as np


def dl_model():
    # Load CIFAR-10 dataset
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Define the input shape
    input_shape = (32, 32, 3)

    # Split the input into three groups along the channel dimension
    def split_input(inputs):
        num_filters = inputs.shape[3] // 3
        split1 = Lambda(lambda x: x[:, :, :, :num_filters])(inputs)
        split2 = Lambda(lambda x: x[:, :, :, num_filters:2*num_filters])(inputs)
        split3 = Lambda(lambda x: x[:, :, :, 2*num_filters:3*num_filters])(inputs)
        return tf.split(split1, 3, axis=-1)

    # Apply 1x1 convolutions to each group
    def conv_layer(x, filters, kernel_size):
        return Conv2D(filters, kernel_size=kernel_size, activation='relu')(x)

    # Downsample each group with average pooling
    def downsample(x):
        return MaxPooling2D(pool_size=2)(x)

    # The main function to build the model
    def build_model():
        # Split input into three groups
        split_x = split_input(tf.concat([train_images, test_images], axis=0))

        # Convolutional layers
        x = tf.concat(split_x, axis=0)
        x = conv_layer(x, filters=len(split_x), kernel_size=1)
        x = downsample(x)
        x = conv_layer(x, filters=len(split_x), kernel_size=1)
        x = downsample(x)
        x = conv_layer(x, filters=len(split_x), kernel_size=1)
        x = downsample(x)

        # Flatten and pass through dense layers
        x = Flatten()(x)
        outputs = Dense(10, activation='softmax')(x)

        # Create the Keras model
        model = Model(inputs=split_input(tf.concat([train_images, test_images], axis=0)), outputs=outputs)
        return model

    return build_model()

# Build and print the model
model = dl_model()
model.summary()