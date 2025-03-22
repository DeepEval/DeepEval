import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 pixels with 3 color channels
    inputs = layers.Input(shape=input_shape)

    # Main pathway with 1x1 convolution
    path1 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(inputs)

    # Parallel branch with 1x1, 1x3, and 3x1 convolutions
    path2_a = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(inputs)
    path2_b = layers.Conv2D(32, (1, 3), padding='same', activation='relu')(inputs)
    path2_c = layers.Conv2D(32, (3, 1), padding='same', activation='relu')(inputs)

    # Concatenating the outputs of the parallel branch
    concatenated = layers.concatenate([path2_a, path2_b, path2_c], axis=-1)

    # 1x1 convolution after concatenation
    main_output = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(concatenated)

    # Additive skip connection from input
    added = layers.add([main_output, path1])

    # Flatten the output for the fully connected layers
    flattened = layers.Flatten()(added)

    # Fully connected layers
    dense1 = layers.Dense(128, activation='relu')(flattened)
    dense2 = layers.Dense(64, activation='relu')(dense1)

    # Output layer for classification (10 classes for CIFAR-10)
    outputs = layers.Dense(10, activation='softmax')(dense2)

    # Constructing the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
model = dl_model()
model.summary()