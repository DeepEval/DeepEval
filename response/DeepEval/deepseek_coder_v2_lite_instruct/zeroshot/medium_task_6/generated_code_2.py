import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Define the model
    model = models.Sequential()

    # Initial convolution layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))

    # Define the three parallel blocks
    for _ in range(3):
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())

    # Add the initial convolution's output to the outputs of the blocks
    added_output = layers.add([model.output, model.get_layer('conv2d').output])

    # Flatten the output
    added_output = layers.Flatten()(added_output)

    # Fully connected layers
    added_output = layers.Dense(128, activation='relu')(added_output)
    added_output = layers.Dropout(0.5)(added_output)
    outputs = layers.Dense(10, activation='softmax')(added_output)

    # Create the final model
    model = models.Model(inputs=model.input, outputs=outputs)

    return model

# Example usage
if __name__ == "__main__":
    model = dl_model()
    model.summary()