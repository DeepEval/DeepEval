import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def dl_model():     

    input_layer = keras.Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    split_layer = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Define the main pathway
    main_pathway = []
    for i in range(3):
        conv1 = layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[i])
        conv2 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        dropout = layers.Dropout(0.2)(conv2)
        main_pathway.append(dropout)
    
    # Concatenate the outputs from the three groups
    concatenated_layer = layers.Concatenate()(main_pathway)
    
    # Define the branch pathway
    branch_pathway = layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine the outputs from both pathways using an addition operation
    combined_layer = layers.Add()([concatenated_layer, branch_pathway])
    
    # Flatten the resulting output
    flatten_layer = layers.Flatten()(combined_layer)
    
    # Add a fully connected layer to complete the classification process
    output_layer = layers.Dense(units=10, activation='softmax')(flatten_layer)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model