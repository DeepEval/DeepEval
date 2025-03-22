import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # First block: Dual-path structure
    # Main path
    main_path = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)
    main_path = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(main_path)
    main_path = layers.Conv2D(3, (3, 3), padding='same')(main_path)  # Restore the number of channels

    # Branch path
    branch_path = layers.Conv2D(3, (1, 1), padding='same')(input_layer)  # Direct connection to input layer

    # Combine paths using addition
    combined_path = layers.add([main_path, branch_path])

    # Second block: Splitting the input into three groups
    split = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(combined_path)

    # Depthwise separable convolutions with different kernel sizes
    conv1 = layers.SeparableConv2D(32, (1, 1), padding='same', activation='relu')(split[0])
    conv2 = layers.SeparableConv2D(32, (3, 3), padding='same', activation='relu')(split[1])
    conv3 = layers.SeparableConv2D(32, (5, 5), padding='same', activation='relu')(split[2])

    # Concatenate the outputs from the three groups
    concatenated = layers.concatenate([conv1, conv2, conv3])

    # Fully connected layers
    flatten = layers.Flatten()(concatenated)
    dense1 = layers.Dense(128, activation='relu')(flatten)
    output_layer = layers.Dense(10, activation='softmax')(dense1)  # CIFAR-10 has 10 classes

    # Create the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of creating the model
model = dl_model()
model.summary()  # Print the model summary to verify architecture