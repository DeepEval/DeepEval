import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)
    
    # Define the input layer
    inputs = keras.Input(shape=input_shape)
    
    # Block 1
    # Split the input into three groups along the last dimension
    split = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    
    # Extract deep features through a series of convolutions
    group1 = layers.Conv2D(6, 1, activation='relu')(split[0])
    group1 = layers.Conv2D(16, 3, activation='relu')(group1)
    group1 = layers.Conv2D(16, 1, activation='relu')(group1)
    
    group2 = layers.Conv2D(6, 1, activation='relu')(split[1])
    group2 = layers.Conv2D(16, 3, activation='relu')(group2)
    group2 = layers.Conv2D(16, 1, activation='relu')(group2)
    
    group3 = layers.Conv2D(6, 1, activation='relu')(split[2])
    group3 = layers.Conv2D(16, 3, activation='relu')(group3)
    group3 = layers.Conv2D(16, 1, activation='relu')(group3)
    
    # Concatenate the outputs from the three groups
    concatenated = layers.Concatenate()([group1, group2, group3])
    
    # Transition convolution layer
    transition = layers.Conv2D(20, 1, activation='relu')(concatenated)
    transition = layers.AveragePooling2D(2, 2)(transition)
    
    # Block 2
    # Global max pooling
    global_max_pool = layers.GlobalMaxPooling2D()(transition)
    
    # Generate channel-matching weights through two fully connected layers
    weights = layers.Dense(10, activation='relu')(global_max_pool)
    weights = layers.Dense(10, activation='relu')(weights)
    
    # Reshape the weights to match the shape of the adjusted output
    weights = layers.Reshape((10, 1, 1))(weights)
    
    # Multiply the weights with the adjusted output
    output_main_path = layers.Multiply()([transition, weights])
    
    # Branch of the model connecting directly to the input
    branch = layers.Conv2D(20, 1, activation='relu')(inputs)
    branch = layers.AveragePooling2D(2, 2)(branch)
    
    # Add the outputs from both the main path and the branch
    added = layers.Add()([output_main_path, branch])
    
    # Final output fed through a fully connected layer for classification
    outputs = layers.Dense(10, activation='softmax')(added)
    
    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model