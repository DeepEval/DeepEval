import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # First block with dual-path structure
    # Main Path
    main_path = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)
    main_path = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(main_path)
    main_path = layers.Conv2D(3, (3, 3), padding='same', activation='relu')(main_path)

    # Branch Path
    branch_path = input_layer

    # Combine paths
    combined = layers.Add()([main_path, branch_path])

    # Second block with depthwise separable convolutions
    # Splitting the channels into 3 groups
    split_groups = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(combined)

    # Depthwise separable convolutions for each group with different kernel sizes
    conv_1x1 = layers.SeparableConv2D(32, (1, 1), padding='same', activation='relu')(split_groups[0])
    conv_3x3 = layers.SeparableConv2D(32, (3, 3), padding='same', activation='relu')(split_groups[1])
    conv_5x5 = layers.SeparableConv2D(32, (5, 5), padding='same', activation='relu')(split_groups[2])

    # Concatenate the outputs from the three groups
    concatenated = layers.Concatenate()([conv_1x1, conv_3x3, conv_5x5])

    # Fully connected layers for classification
    flatten = layers.Flatten()(concatenated)
    dense_1 = layers.Dense(128, activation='relu')(flatten)
    output_layer = layers.Dense(10, activation='softmax')(dense_1)  # CIFAR-10 has 10 classes

    # Create the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of how to create the model
model = dl_model()
model.summary()  # To see the architecture of the model