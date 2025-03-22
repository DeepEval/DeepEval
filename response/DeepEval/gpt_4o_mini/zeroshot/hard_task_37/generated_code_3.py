import tensorflow as tf
from tensorflow.keras import layers, models

def create_parallel_branch(input_tensor):
    # Define the block with three sequential convolutional layers
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    
    # Splitting the outputs into separate paths
    path1 = layers.GlobalAveragePooling2D()(x)
    path2 = layers.GlobalMaxPooling2D()(x)
    
    return path1, path2

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(28, 28, 1))

    # First branch
    branch1 = create_parallel_branch(input_layer)

    # Second branch (parallel)
    conv_branch = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    branch2 = create_parallel_branch(conv_branch)

    # Combine the outputs of both branches using addition
    combined_output1 = layers.add(branch1)
    combined_output2 = layers.add(branch2)

    # Concatenate outputs from both blocks
    concatenated = layers.concatenate([combined_output1, combined_output2])

    # Flatten and add a fully connected layer
    flatten_output = layers.Flatten()(concatenated)
    dense_output = layers.Dense(128, activation='relu')(flatten_output)

    # Output layer
    output_layer = layers.Dense(10, activation='softmax')(dense_output)

    # Construct the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of how to use the function to create the model
model = dl_model()
model.summary()