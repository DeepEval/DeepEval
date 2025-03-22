import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    inputs = layers.Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    group_size = inputs.shape[-1] // 3
    groups = tf.split(inputs, num_or_size_splits=3, axis=-1)

    # Apply 1x1 convolutions to each group
    conv_outputs = []
    for i, group in enumerate(groups):
        filters = group_size if i == 0 else group_size * 2
        conv_outputs.append(layers.Conv2D(filters=filters, kernel_size=(1, 1), activation='relu')(group))

    # Downsample each group using average pooling
    pool_outputs = []
    for conv_output in conv_outputs:
        pool_outputs.append(layers.AveragePooling2D(pool_size=(2, 2), strides=2)(conv_output))

    # Concatenate the resulting groups
    concat_output = layers.concatenate(pool_outputs)

    # Flatten and pass through fully connected layers
    flattened = layers.Flatten()(concat_output)
    dense1 = layers.Dense(units=64, activation='relu')(flattened)
    dense2 = layers.Dense(units=32, activation='relu')(dense1)

    # Output layer
    outputs = layers.Dense(units=10, activation='softmax')(dense2)

    # Create and return the model
    model = models.Model(inputs=inputs, outputs=outputs)
    return model