import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def dl_model():
    
    # Define the input layer
    input_layer = keras.Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    def split_input(input_tensor):
        input_channels = input_tensor.shape[-1]
        num_groups = 3
        split_input = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=num_groups, axis=-1))(input_tensor)
        return split_input
    
    split_output = split_input(input_layer)
    
    # Apply 1x1 convolutions to each group independently
    def conv1x1(input_tensor, filters):
        return layers.Conv2D(filters=filters, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
    
    conv1_group1 = conv1x1(split_output[0], filters=int(32/3))
    conv1_group2 = conv1x1(split_output[1], filters=int(32/3))
    conv1_group3 = conv1x1(split_output[2], filters=int(32/3))
    
    # Apply average pooling to each group
    def avg_pooling(input_tensor):
        return layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_tensor)
    
    avg_pool_group1 = avg_pooling(conv1_group1)
    avg_pool_group2 = avg_pooling(conv1_group2)
    avg_pool_group3 = avg_pooling(conv1_group3)
    
    # Concatenate the three groups along the channel dimension
    concat_output = layers.Concatenate()([avg_pool_group1, avg_pool_group2, avg_pool_group3])
    
    # Flatten the concatenated feature maps into a one-dimensional vector
    flatten_layer = layers.Flatten()(concat_output)
    
    # Apply two fully connected layers for classification
    dense1 = layers.Dense(64, activation='relu')(flatten_layer)
    output_layer = layers.Dense(10, activation='softmax')(dense1)
    
    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model