import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    model = models.Sequential()

    # Initial convolutional layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # First parallel path with global average pooling and two fully connected layers
    avg_path = models.Sequential()
    avg_path.add(layers.GlobalAveragePooling2D())
    avg_path.add(layers.Dense(128, activation='relu'))
    avg_path.add(layers.Dense(64, activation='relu'))

    # Second parallel path with global max pooling and two fully connected layers
    max_path = models.Sequential()
    max_path.add(layers.GlobalMaxPooling2D())
    max_path.add(layers.Dense(128, activation='relu'))
    max_path.add(layers.Dense(64, activation='relu'))

    # Concatenate the outputs of the two paths
    combined = layers.add([avg_path(model.output), max_path(model.output)])
    combined = layers.Activation('relu')(combined)

    # Extract spatial features
    spatial_features = layers.Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2])(combined))
    spatial_features = layers.Reshape((1, 1, 128))(spatial_features)

    # Concatenate channel and spatial features
    fused_features = layers.concatenate([combined, spatial_features], axis=-1)

    # Flatten the fused features
    flattened_features = layers.Flatten()(fused_features)

    # Fully connected layer for final output
    output = layers.Dense(10, activation='softmax')(flattened_features)

    # Construct the final model
    model = models.Model(inputs=model.input, outputs=output)

    return model

# Example usage:
# model = dl_model()
# model.summary()