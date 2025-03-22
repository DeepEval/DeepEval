import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    # Define the parallel branches

    # Block 1
    block1_input = layers.Input(shape=(28, 28, 1))
    block1_conv1 = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(block1_input)
    block1_conv2 = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(block1_conv1)
    block1_conv3 = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(block1_conv2)
    block1_output = layers.MaxPooling2D()(block1_conv3)

    # Block 2
    block2_input = layers.Input(shape=(28, 28, 1))
    block2_conv1 = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(block2_input)
    block2_conv2 = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(block2_conv1)
    block2_conv3 = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(block2_conv2)
    block2_output = layers.MaxPooling2D()(block2_conv3)

    # Parallel branch
    parallel_input = layers.Input(shape=(28, 28, 1))
    parallel_conv = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(parallel_input)
    parallel_output = layers.MaxPooling2D()(parallel_conv)

    # Combine the outputs from the parallel branches
    combined_output = layers.Add()([block1_output, block2_output, parallel_output])

    # Concatenate the outputs from the two blocks
    merged_output = layers.concatenate([combined_output, combined_output])

    # Flatten and fully connected layers
    flattened_output = layers.Flatten()(merged_output)
    fully_connected_output = layers.Dense(units=10, activation='softmax')(flattened_output)

    # Create the model
    model = tf.keras.Model(
        inputs=[block1_input, block2_input, parallel_input],
        outputs=fully_connected_output
    )

    return model