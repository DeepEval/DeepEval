import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, DepthwiseConv2D, BatchNormalization, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model


def dl_model():
    
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the number of classes
    num_classes = 10

    # Define the split size for the first block
    split_size = 3

    # Define the kernel sizes for the depthwise separable convolutional layers
    kernel_sizes = [1, 3, 5]

    # Define the activation function for the convolutional layers
    activation = 'relu'

    # Define the batch normalization axis
    batch_axis = -1

    # Define the input layer
    input_layer = Input(shape=input_shape)

    # Define the first block
    x = Lambda(lambda x: tf.split(x, split_size, axis=-1))(input_layer)
    x = [Conv2D(32, kernel_sizes[i], activation=activation, padding='same')(x[i]) for i in range(split_size)]
    x = [BatchNormalization(axis=batch_axis)(x[i]) for i in range(split_size)]
    x = Concatenate()(x)

    # Define the second block
    x = Lambda(lambda x: tf.split(x, split_size, axis=-1))(x)
    x = [Conv2D(32, kernel_sizes[i], activation=activation, padding='same')(x[i]) for i in range(split_size)]
    x = [BatchNormalization(axis=batch_axis)(x[i]) for i in range(split_size)]
    x = Concatenate()(x)

    # Define the third block
    x = Lambda(lambda x: tf.split(x, split_size, axis=-1))(x)
    x = [Conv2D(32, kernel_sizes[i], activation=activation, padding='same')(x[i]) for i in range(split_size)]
    x = [BatchNormalization(axis=batch_axis)(x[i]) for i in range(split_size)]
    x = Concatenate()(x)

    # Define the fourth block
    x = Lambda(lambda x: tf.split(x, split_size, axis=-1))(x)
    x = [Conv2D(32, kernel_sizes[i], activation=activation, padding='same')(x[i]) for i in range(split_size)]
    x = [BatchNormalization(axis=batch_axis)(x[i]) for i in range(split_size)]
    x = Concatenate()(x)

    # Define the fifth block
    x = Lambda(lambda x: tf.split(x, split_size, axis=-1))(x)
    x = [Conv2D(32, kernel_sizes[i], activation=activation, padding='same')(x[i]) for i in range(split_size)]
    x = [BatchNormalization(axis=batch_axis)(x[i]) for i in range(split_size)]
    x = Concatenate()(x)

    # Define the final block
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)

    # Define the model
    model = Model(inputs=input_layer, outputs=x)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model