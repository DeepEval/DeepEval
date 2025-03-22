import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    
    # Define the input shape
    input_shape = (32, 32, 3)
    
    # Add input layer
    input_layer = keras.Input(shape=input_shape)
    
    # Split the input into three groups along the channel dimension
    split_layer = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Define the main pathway
    main_pathway = layers.Conv2D(32, (1, 1), activation='relu')(split_layer[0])
    main_pathway = layers.Conv2D(32, (3, 3), activation='relu')(main_pathway)
    main_pathway = layers.Dropout(0.2)(main_pathway)
    
    main_pathway2 = layers.Conv2D(32, (1, 1), activation='relu')(split_layer[1])
    main_pathway2 = layers.Conv2D(32, (3, 3), activation='relu')(main_pathway2)
    main_pathway2 = layers.Dropout(0.2)(main_pathway2)
    
    main_pathway3 = layers.Conv2D(32, (1, 1), activation='relu')(split_layer[2])
    main_pathway3 = layers.Conv2D(32, (3, 3), activation='relu')(main_pathway3)
    main_pathway3 = layers.Dropout(0.2)(main_pathway3)
    
    # Concatenate the outputs from the three groups
    main_output = layers.Concatenate()([main_pathway, main_pathway2, main_pathway3])
    
    # Define the branch pathway
    branch_pathway = layers.Conv2D(32, (1, 1), activation='relu')(input_layer)
    
    # Combine the outputs from both pathways
    output = layers.Add()([main_output, branch_pathway])
    
    # Flatten the output
    flatten_layer = layers.Flatten()(output)
    
    # Add a dense layer
    dense_layer = layers.Dense(64, activation='relu')(flatten_layer)
    
    # Add a dropout layer
    dropout_layer = layers.Dropout(0.2)(dense_layer)
    
    # Add the output layer
    output_layer = layers.Dense(10, activation='softmax')(dropout_layer)
    
    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model