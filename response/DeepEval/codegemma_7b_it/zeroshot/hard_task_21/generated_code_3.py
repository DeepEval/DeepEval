import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    inputs = layers.Input((32, 32, 3))

    # Main path
    main_path = inputs
    main_path = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(main_path)
    main_path = [
        layers.SeparableConv2D(32, (1, 1), activation='relu', padding='same')(branch)
        for branch in main_path
    ]
    main_path = layers.Concatenate()(main_path)

    # Branch path
    branch_path = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(inputs)

    # Concatenate main and branch outputs
    concat_path = layers.Add()([main_path, branch_path])

    # Fully connected layers
    x = layers.GlobalAveragePooling2D()(concat_path)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model