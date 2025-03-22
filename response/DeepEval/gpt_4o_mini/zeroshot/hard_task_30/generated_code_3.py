import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # First block: Dual-path structure
    # Main path
    main_path = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)
    main_path = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(main_path)
    main_path = layers.Conv2D(3, (3, 3), padding='same', activation='relu')(main_path)  # Restore number of channels

    # Branch path
    branch_path = input_layer

    # Combine paths
    combined = layers.Add()([main_path, branch_path])

    # Second block: Split and depthwise separable convolutions
    split = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(combined)

    # Depthwise separable convolutions with different kernel sizes
    conv1x1 = layers.SeparableConv2D(32, (1, 1), padding='same', activation='relu')(split[0])
    conv3x3 = layers.SeparableConv2D(32, (3, 3), padding='same', activation='relu')(split[1])
    conv5x5 = layers.SeparableConv2D(32, (5, 5), padding='same', activation='relu')(split[2])

    # Concatenation of the outputs
    concatenated = layers.Concatenate()([conv1x1, conv3x3, conv5x5])

    # Fully connected layers
    flatten = layers.Flatten()(concatenated)
    dense1 = layers.Dense(128, activation='relu')(flatten)
    output_layer = layers.Dense(10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    # Construct the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model