import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    
    # Define the input shape for the MNIST dataset
    input_shape = (28, 28, 1) 

    # Define the specialized block
    def specialized_block(inputs):
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
        x = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
        x = layers.Conv2D(32, (1, 1), activation='relu')(x)
        x = layers.Dropout(0.25)(x)
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
    outputs = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = tf.keras.Model(inputs=[branch1_input, branch2_input, branch3_input], outputs=outputs)

    return model