import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.datasets import cifar10

def dl_model():
    # Input layer for CIFAR-10 images (32x32x3)
    input_layer = layers.Input(shape=(32, 32, 3))
    
    # First block with three max pooling layers of different scales
    pool_1 = layers.MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    pool_2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    pool_3 = layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)
    
    # Flatten the pooling outputs
    flatten_1 = layers.Flatten()(pool_1)
    flatten_2 = layers.Flatten()(pool_2)
    flatten_3 = layers.Flatten()(pool_3)
    
    # Apply dropout to mitigate overfitting
    dropout_rate = 0.5
    dropout_1 = layers.Dropout(dropout_rate)(flatten_1)
    dropout_2 = layers.Dropout(dropout_rate)(flatten_2)
    dropout_3 = layers.Dropout(dropout_rate)(flatten_3)

    # Concatenate the flattened outputs
    concatenated = layers.Concatenate()([dropout_1, dropout_2, dropout_3])
    
    # Fully connected layer
    dense_layer = layers.Dense(256, activation='relu')(concatenated)
    
    # Reshape the output into a 4D tensor
    reshaped = layers.Reshape((4, 4, 16))(dense_layer)  # Assuming the output size can be reshaped like this
    
    # Second block: split and process through separable convolutions
    split_tensors = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=-1))(reshaped)
    
    # Apply separable convolutions with different kernel sizes
    conv_outputs = []
    for kernel_size in [1, 3, 5, 7]:
        conv_output = layers.SeparableConv2D(filters=16, kernel_size=(kernel_size, kernel_size), padding='same', activation='relu')(split_tensors[0])
        conv_outputs.append(conv_output)
    
    # Concatenate the outputs from the separable convolutions
    concatenated_conv = layers.Concatenate()(conv_outputs)
    
    # Flatten and add a final dense layer for classification
    flatten_conv = layers.Flatten()(concatenated_conv)
    output_layer = layers.Dense(10, activation='softmax')(flatten_conv)  # 10 classes for CIFAR-10

    # Construct the model
    model = models.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate the model
model = dl_model()
model.summary()