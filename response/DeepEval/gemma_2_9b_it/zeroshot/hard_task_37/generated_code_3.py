import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    
    # Define the block
    def block(input_tensor):
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)
        x1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x1)
        
        # Parallel branch
        p_branch = layers.Conv2D(128, (1, 1), activation='relu')(input_tensor)

        # Combine paths
        x = layers.Add()([x2, p_branch])

        return x

    # Input layer
    inputs = keras.Input(shape=(28, 28, 1))

    # First block
    branch1 = block(inputs)

    # Second block
    branch2 = block(branch1)

    # Concatenate outputs
    x = layers.Concatenate()([branch1, branch2])

    # Flatten and fully connected layer
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model