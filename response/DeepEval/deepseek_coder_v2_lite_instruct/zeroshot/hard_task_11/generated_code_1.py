import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate

def dl_model():
    input_shape = (32, 32, 3)
    
    # Define the input layer
    inputs = Input(shape=input_shape)
    
    # Main pathway
    main_pathway = Conv2D(32, (1, 1), activation='relu')(inputs)
    main_pathway = Conv2D(32, (1, 3), activation='relu', padding='same')(main_pathway)
    main_pathway = Conv2D(32, (3, 1), activation='relu', padding='same')(main_pathway)
    main_pathway = Conv2D(32, (1, 1), activation='relu')(main_pathway)
    
    # Parallel branch
    parallel_branch = Conv2D(32, (1, 1), activation='relu')(inputs)
    parallel_branch = Conv2D(32, (1, 3), activation='relu', padding='same')(parallel_branch)
    parallel_branch = Conv2D(32, (3, 1), activation='relu', padding='same')(parallel_branch)
    parallel_branch = Conv2D(32, (1, 1), activation='relu')(parallel_branch)
    
    # Concatenate the outputs of the main pathway and the parallel branch
    concatenated = concatenate([main_pathway, parallel_branch])
    
    # Add a 1x1 convolution to produce the main output
    output = Conv2D(32, (1, 1), activation='relu')(concatenated)
    
    # Direct connection from the input to the model's branch
    direct_connection = inputs
    
    # Add the direct connection and the main pathway via an additive operation
    final_output = tf.add(direct_connection, output)
    
    # Flatten the final output
    final_output = Flatten()(final_output)
    
    # Fully connected layers for classification
    final_output = Dense(128, activation='relu')(final_output)
    final_output = Dense(10, activation='softmax')(final_output)
    
    # Define the model
    model = Model(inputs=inputs, outputs=final_output)
    
    return model

# Example usage:
# model = dl_model()
# model.summary()