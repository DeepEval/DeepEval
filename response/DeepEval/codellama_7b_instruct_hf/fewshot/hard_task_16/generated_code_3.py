import keras
import tensorflow as tf

def dl_model():
    # Input layer
    input_layer = keras.layers.Input(shape=(32, 32, 3))

    # First block
    x = input_layer
    x = keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    # Split output into three groups
    x = keras.layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(x)

    # Convolutional layers and pooling layers for each group
    for i in range(3):
        x[i] = keras.layers.Conv2D(64, (3, 3), activation='relu')(x[i])
        x[i] = keras.layers.MaxPooling2D((2, 2))(x[i])

    # Concatenate outputs from three groups
    x = keras.layers.Concatenate()(x)

    # Transition convolution layer to adjust number of channels
    x = keras.layers.Conv2D(128, (1, 1), activation='relu')(x)

    # Second block
    x = keras.layers.GlobalMaxPooling2D()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dense(10)(x)

    # Add main path and branch path
    x = keras.layers.Add()([x, input_layer])

    # Flatten output and pass through fully connected layer for classification
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(10)(x)

    # Return constructed model
    model = keras.Model(inputs=input_layer, outputs=x)
    return model