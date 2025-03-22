import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

def dl_model():
    # Input Layer
    inputs = layers.Input(shape=(32, 32, 3))

    # Block 1: Splitting the input into three groups
    split_inputs = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)

    # Convolutional operations for each split
    def conv_block(x):
        x = layers.Conv2D(64, (1, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(64, (1, 1), activation='relu', padding='same')(x)
        return x

    # Apply conv_block to each split input
    conv_outputs = [conv_block(split) for split in split_inputs]

    # Concatenate outputs from the three groups
    block1_output = layers.concatenate(conv_outputs)

    # Transition Convolution
    transition_conv = layers.Conv2D(192, (1, 1), padding='same')(block1_output)

    # Block 2: Global Max Pooling
    global_pool = layers.GlobalMaxPooling2D()(transition_conv)

    # Fully connected layers to generate channel-matching weights
    dense1 = layers.Dense(128, activation='relu')(global_pool)
    dense2 = layers.Dense(transition_conv.shape[-1], activation='sigmoid')(dense1)

    # Reshape weights to match the shape of transition_conv
    reshaped_weights = layers.Reshape((1, 1, transition_conv.shape[-1]))(dense2)

    # Main path output
    main_path_output = layers.multiply([transition_conv, reshaped_weights])

    # Branch connecting directly to input
    branch_output = inputs

    # Combine main path and branch
    combined_output = layers.add([main_path_output, branch_output])

    # Final output layer for classification
    final_output = layers.Flatten()(combined_output)
    final_output = layers.Dense(10, activation='softmax')(final_output)  # 10 classes for CIFAR-10

    # Create the model
    model = models.Model(inputs=inputs, outputs=final_output)

    return model

# Example of how to create the model
model = dl_model()
model.summary()