import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input layer
    input_layer = keras.Input(shape=(32, 32, 3))

    # Main path: depthwise separable convolutional layers with varying kernel sizes
    def main_path(input_tensor):
        split_tensor = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        conv1x1 = layers.SeparableConv2D(32, (1, 1), activation='relu')(split_tensor[0])
        conv3x3 = layers.SeparableConv2D(32, (3, 3), activation='relu')(split_tensor[1])
        conv5x5 = layers.SeparableConv2D(32, (5, 5), activation='relu')(split_tensor[2])

        # Concatenate the outputs from the three groups
        output_tensor = layers.Concatenate()([conv1x1, conv3x3, conv5x5])

        return output_tensor

    main_path_output = main_path(input_layer)

    # Branch path: 1x1 convolutional layer to align the number of output channels
    branch_path_output = layers.SeparableConv2D(32, (1, 1), activation='relu')(input_layer)

    # Add the outputs from the main and branch paths
    added_output = layers.Add()([main_path_output, branch_path_output])

    # Flatten the output
    flatten_layer = layers.Flatten()(added_output)

    # Define the two fully connected layers
    dense1 = layers.Dense(64, activation='relu')(flatten_layer)
    output_layer = layers.Dense(10, activation='softmax')(dense1)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model