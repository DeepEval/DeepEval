import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    
    # Define the input shape for the MNIST dataset
    input_shape = (28, 28, 1) 

    # Define the specialized block
    def specialized_block(input_tensor):
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input_tensor)
        x = layers.Dropout(0.2)(x)
        x = layers.Conv2D(64, (1, 1), activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        return x

    # Create the three branches
    branch1_output = specialized_block(layers.Input(shape=input_shape))
    branch2_output = specialized_block(layers.Input(shape=input_shape))
    branch3_output = specialized_block(layers.Input(shape=input_shape))

    # Concatenate the outputs from the branches
    merged_output = layers.concatenate([branch1_output, branch2_output, branch3_output], axis=-1)

    # Flatten the concatenated output
    x = layers.Flatten()(merged_output)

    # Add two fully connected layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = tf.keras.Model(inputs=[branch1_input, branch2_input, branch3_input], outputs=output)
    
    return model 

# Get the model
model = dl_model()