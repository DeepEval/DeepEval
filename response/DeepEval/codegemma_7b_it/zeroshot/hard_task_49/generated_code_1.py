import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():

    # Input layer
    inputs = keras.Input(shape=(28, 28, 1))

    # First block
    x = inputs
    x = layers.AveragePooling2D(pool_size=1, strides=1, padding='valid')(x)
    x = layers.AveragePooling2D(pool_size=2, strides=2, padding='valid')(x)
    x = layers.AveragePooling2D(pool_size=4, strides=4, padding='valid')(x)
    x = layers.Flatten()(x)

    # Second block
    x = layers.Reshape((4, 4, 1))(x)
    split_outputs = tf.split(x, 4, axis=-1)
    results = []
    for idx, split_output in enumerate(split_outputs):
        split_output = layers.DepthwiseConv2D(
            kernel_size=(1, 1) if idx == 0 else (3, 3) if idx == 1 else (5, 5) if idx == 2 else (7, 7),
            strides=1, padding='valid')(split_output)
        results.append(split_output)

    # Concatenate results and flatten
    concat_outputs = layers.concatenate(results)
    concat_outputs = layers.Flatten()(concat_outputs)

    # Fully connected layer and output
    outputs = layers.Dense(10, activation='softmax')(concat_outputs)

    # Create and return the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Create and print the model summary
model = dl_model()
model.summary()