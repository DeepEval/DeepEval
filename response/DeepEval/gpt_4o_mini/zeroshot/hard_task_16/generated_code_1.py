import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_tensor = layers.Input(shape=(32, 32, 3))

    # Block 1: Splitting input into three groups
    split_inputs = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)

    def create_branch(input_tensor):
        x = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(input_tensor)
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(x)
        return x

    # Process each branch
    branch_outputs = [create_branch(branch) for branch in split_inputs]
    
    # Concatenate the outputs of the branches
    concatenated = layers.Concatenate(axis=-1)(branch_outputs)

    # Transition Convolution
    transition_conv = layers.Conv2D(96, (1, 1), padding='same', activation='relu')(concatenated)

    # Block 2: Global Max Pooling
    global_pool = layers.GlobalMaxPooling2D()(transition_conv)

    # Fully connected layers for channel-matching weights
    fc1 = layers.Dense(128, activation='relu')(global_pool)
    fc2 = layers.Dense(96, activation='sigmoid')(fc1)  # Match the number of channels from transition_conv

    # Reshape weights to match the adjusted output shape
    reshaped_weights = layers.Reshape((1, 1, 96))(fc2)

    # Main path output multiplication
    main_path_output = layers.multiply([transition_conv, reshaped_weights])

    # Branch connection directly to input
    branch_output = layers.Conv2D(96, (1, 1), padding='same', activation='relu')(input_tensor)

    # Combine main path output and branch output
    combined_output = layers.Add()([main_path_output, branch_output])

    # Final classification layer
    flatten_output = layers.Flatten()(combined_output)
    final_output = layers.Dense(10, activation='softmax')(flatten_output)  # CIFAR-10 has 10 classes

    # Model construction
    model = models.Model(inputs=input_tensor, outputs=final_output)

    return model

# Example usage
# model = dl_model()
# model.summary()