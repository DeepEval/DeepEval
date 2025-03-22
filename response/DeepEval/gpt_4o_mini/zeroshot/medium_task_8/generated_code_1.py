import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

def dl_model():
    # Input layer
    input_shape = (32, 32, 3)
    inputs = layers.Input(shape=input_shape)

    # Main Path
    # Split the input into three groups along the last dimension
    split_inputs = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(inputs)
    
    group1 = split_inputs[0]  # First group remains unchanged
    group2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(split_inputs[1])  # Second group
    group3 = split_inputs[2]  # Third group
    
    # Combine second and third groups
    combined_group = layers.Concatenate()([group2, group3])
    
    # Additional 3x3 convolution on combined group
    main_path_output = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(combined_group)

    # Concatenate all three groups
    main_path_output = layers.Concatenate()([group1, main_path_output])

    # Branch Path
    branch_path_output = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(inputs)

    # Fuse main and branch paths together
    fused_output = layers.Add()([main_path_output, branch_path_output])

    # Flatten and fully connected layer for classification
    flattened_output = layers.Flatten()(fused_output)
    dense_output = layers.Dense(10, activation='softmax')(flattened_output)  # 10 classes for CIFAR-10

    # Create the model
    model = models.Model(inputs=inputs, outputs=dense_output)

    return model

# Example usage
model = dl_model()
model.summary()