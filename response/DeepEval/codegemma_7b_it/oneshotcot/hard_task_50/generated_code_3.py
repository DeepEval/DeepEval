import keras
from tensorflow.keras import layers

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # First block
    x = layers.MaxPooling2D(pool_size=1, strides=1, padding='same')(input_layer)
    x = layers.MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
    x = layers.MaxPooling2D(pool_size=4, strides=4, padding='same')(x)
    x = layers.concatenate([layers.Flatten()(x)])

    # Dropout to mitigate overfitting
    x = layers.Dropout(rate=0.4)(x)

    # Reshape for second block
    x = layers.Reshape(target_shape=[-1, 4, 4, 1])(x)

    # Second block
    x = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=-1))(x)

    # Separable convolutional layers
    path_1 = layers.SeparableConv2D(filters=64, kernel_size=1, padding='same')(x[0])
    path_2 = layers.SeparableConv2D(filters=64, kernel_size=3, padding='same')(x[1])
    path_3 = layers.SeparableConv2D(filters=64, kernel_size=5, padding='same')(x[2])
    path_4 = layers.SeparableConv2D(filters=64, kernel_size=7, padding='same')(x[3])

    # Concatenate outputs
    x = layers.concatenate([path_1, path_2, path_3, path_4])

    # Flatten and fully connected layer
    x = layers.Flatten()(x)
    output_layer = layers.Dense(units=10, activation='softmax')(x)

    # Model creation
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model