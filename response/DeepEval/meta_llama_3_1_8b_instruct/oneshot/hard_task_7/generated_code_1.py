import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Initial convolutional layer with 32 kernels
    initial_conv = layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape)

    # Define Block 1
    def block1(input_tensor):
        # Split the input into two groups along the last dimension
        input_groups = layers.Lambda(lambda x: tf.split(x, 2, axis=-1))(input_tensor)

        # Process the first group
        conv1 = layers.Conv2D(32, (1, 1), activation='relu')(input_groups[0])
        depthwise_conv = layers.DepthwiseConv2D((3, 3), activation='relu')(conv1)
        conv2 = layers.Conv2D(32, (1, 1), activation='relu')(depthwise_conv)

        # The second group is passed through without modification
        merged_output = layers.Concatenate()([conv2, input_groups[1]])

        return merged_output

    # Apply Block 1
    block1_output = block1(initial_conv.output)

    # Define Block 2
    def block2(input_tensor):
        # Get the shape of the input
        input_shape = tf.shape(input_tensor)

        # Reshape the input into four groups
        reshaped_input = layers.Reshape((input_shape[1], input_shape[2], input_shape[3]//2, 2))(input_tensor)

        # Swap the third and fourth dimensions
        swapped_input = layers.Permute((1, 2, 4, 3))(reshaped_input)

        # Reshape the input back to its original shape
        reshaped_output = layers.Reshape((input_shape[1], input_shape[2], input_shape[3]))(swapped_input)

        return reshaped_output

    # Apply Block 2
    block2_output = block2(block1_output)

    # Flatten the output and pass it through a fully connected layer
    flatten_layer = layers.Flatten()(block2_output)
    dense_layer = layers.Dense(10, activation='softmax')(flatten_layer)

    # Define the model
    model = keras.Model(inputs=initial_conv.input, outputs=dense_layer)

    return model