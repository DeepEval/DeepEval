import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_tensor = keras.Input(shape=(28, 28, 1))

    # Define the block
    def block(input_tensor):
        x = layers.Conv2D(32, (3, 3), activation='relu')(input_tensor)
        x1 = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x2 = layers.Conv2D(128, (3, 3), activation='relu')(x)

        # Parallel branch
        x3 = layers.Conv2D(64, (1, 1), activation='relu')(input_tensor)

        # Combine outputs
        x = layers.Add()([x1, x2, x3])
        return x

    # Create the two parallel branches
    branch1 = block(input_tensor)
    branch2 = block(input_tensor)

    # Concatenate the outputs
    x = layers.Concatenate()([branch1, branch2])

    # Flatten and fully connected layer
    x = layers.Flatten()(x)
    output_tensor = layers.Dense(10, activation='softmax')(x)

    # Construct the model
    model = keras.Model(inputs=input_tensor, outputs=output_tensor)

    return model