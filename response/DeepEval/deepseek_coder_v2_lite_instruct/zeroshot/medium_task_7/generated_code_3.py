import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))

    # Add outputs of the first two convolutional layers with the output of the third convolutional layer
    add1 = layers.Add()([model.output, model.layers[-1].output])

    # Separate convolutional layer processing the input directly
    separate_conv = layers.Conv2D(64, (3, 3), activation='relu')(model.input)

    # Add the outputs from all paths
    add2 = layers.Add()([add1, separate_conv])

    # Flatten the output
    flatten = layers.Flatten()(add2)

    # Fully connected layers
    dense1 = layers.Dense(128, activation='relu')(flatten)
    dense2 = layers.Dense(10, activation='softmax')(dense1)

    # Construct the final model
    model = models.Model(inputs=model.input, outputs=dense2)

    return model

# Example usage:
# model = dl_model()
# model.summary()