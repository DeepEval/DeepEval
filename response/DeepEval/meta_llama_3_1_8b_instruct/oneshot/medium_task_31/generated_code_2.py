import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input shape of the CIFAR-10 dataset
    input_shape = (32, 32, 3)
    
    # Define the input layer
    input_layer = keras.Input(shape=input_shape)
    
    # Split the input into three groups along the channel dimension
    split_layer = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    # Apply different convolutional kernels to each group
    conv1x1 = layers.Conv2D(64, kernel_size=1, strides=1, padding='same')(split_layer[0])
    conv3x3 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same')(split_layer[1])
    conv5x5 = layers.Conv2D(64, kernel_size=5, strides=1, padding='same')(split_layer[2])
    
    # Concatenate the outputs from the three groups
    output_tensor = layers.Concatenate()([conv1x1, conv3x3, conv5x5])
    
    # Apply batch normalization and ReLU activation
    batch_norm = layers.BatchNormalization()(output_tensor)
    relu = layers.ReLU()(batch_norm)
    
    # Flatten the output
    flatten_layer = layers.Flatten()(relu)
    
    # Apply two fully connected layers
    dense1 = layers.Dense(128, activation='relu')(flatten_layer)
    output_layer = layers.Dense(10, activation='softmax')(dense1)
    
    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model