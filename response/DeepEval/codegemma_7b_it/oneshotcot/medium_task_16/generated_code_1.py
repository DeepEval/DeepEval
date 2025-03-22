import keras
from keras import layers
from keras.models import Model

def dl_model():
    # Define input layer
    inputs = keras.Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    x = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(inputs)

    # Apply 1x1 convolutions to each group independently
    x = [layers.Conv2D(filters=10, kernel_size=(1, 1), padding="same", activation="relu")(group) for group in x]

    # Downsample each group using average pooling
    x = [layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(group) for group in x]

    # Concatenate the three groups along the channel dimension
    x = layers.Concatenate(axis=3)(x)

    # Flatten the concatenated feature maps
    x = layers.Flatten()(x)

    # Pass through two fully connected layers for classification
    x = layers.Dense(units=64, activation="relu")(x)
    outputs = layers.Dense(units=10, activation="softmax")(x)

    # Construct model
    model = Model(inputs=inputs, outputs=outputs)

    return model