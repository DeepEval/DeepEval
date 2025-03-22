import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_layer = keras.Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    split_input = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    # Define the main pathway
    def main_pathway(input_tensor):
        conv1 = layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        drop = layers.Dropout(0.2)(conv2)
        return drop
    
    # Create the main pathway for the three groups
    main_pathway_output = [main_pathway(group) for group in split_input]
    
    # Concatenate the outputs from the three groups
    concatenated_output = layers.Concatenate()(main_pathway_output)
    
    # Define the branch pathway
    branch_input = layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine the outputs from both pathways using addition
    combined_output = layers.Add()([concatenated_output, branch_input])
    
    # Flatten the output
    flatten_layer = layers.Flatten()(combined_output)
    
    # Add a fully connected layer for classification
    output_layer = layers.Dense(units=10, activation='softmax')(flatten_layer)
    
    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model