import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, MaxPooling2D, Flatten, Dense, Lambda, Concatenate, Conv2D, SeparableConv2D
from tensorflow.keras.optimizers import Adam
import numpy as np



def dl_model():

    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Define the input shape
    input_shape = (32, 32, 3)
    input_tensor = Input(shape=input_shape)

    # First Block: Max Pooling Layers and Dropout
    x = input_tensor
    for _ in range(3):
        x = MaxPooling2D((1, 1), strides=(1, 1), padding='valid')(x)
        x = Flatten()(x)
        x = Dropout(0.5)(x)

    # Second Block: Convolutional Layers and Split
    x = Dense(512)(x)
    x = Activation('relu')(x)
    x = Dense(512)(x)
    x = Activation('relu')(x)

    # Split the tensor into four groups
    split_dim = 4
    split_tensor = Lambda(lambda tensors: tf.split(tensors, split_dim, axis=-1))(x)

    # Process each group with separable convolutions
    groups = [
        SeparableConv2D(64, (1, 1))(split_tensor[0]),
        SeparableConv2D(64, (3, 3))(split_tensor[1]),
        SeparableConv2D(64, (5, 5))(split_tensor[2]),
        SeparableConv2D(64, (7, 7))(split_tensor[3])
    ]

    # Concatenate the outputs
    concatenate_tensor = Concatenate()(groups)

    # Fully connected layer and output layer
    output = Dense(10)(concatenate_tensor)

    # Model
    model = Model(inputs=input_tensor, outputs=output)

    # Compile the model
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Return the model
    return model

# Call the function to create the model
model = dl_model()