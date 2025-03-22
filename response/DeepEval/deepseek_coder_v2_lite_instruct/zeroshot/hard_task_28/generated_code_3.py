import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Main path
    x = layers.Input(shape=input_shape)
    
    # Depthwise convolution
    main_path = layers.DepthwiseConv2D(kernel_size=(7, 7), padding='same')(x)
    main_path = layers.LayerNormalization()(main_path)
    
    # Pointwise convolution layers
    main_path = layers.Conv2D(filters=main_path.shape[-1], kernel_size=(1, 1), padding='same')(main_path)
    main_path = layers.Conv2D(filters=main_path.shape[-1], kernel_size=(1, 1), padding='same')(main_path)
    
    # Branch path
    branch_path = x
    
    # Add the outputs of both paths
    combined_output = layers.Add()([main_path, branch_path])
    
    # Flatten the combined output
    flattened_output = layers.Flatten()(combined_output)
    
    # Fully connected layers
    fc1 = layers.Dense(128, activation='relu')(flattened_output)
    fc2 = layers.Dense(10, activation='softmax')(fc1)
    
    # Construct the model
    model = models.Model(inputs=x, outputs=fc2)
    
    return model

# Example usage:
# model = dl_model()
# model.summary()