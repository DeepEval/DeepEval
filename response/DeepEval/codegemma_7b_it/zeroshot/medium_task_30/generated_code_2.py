import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    # Create the input layer
    inputs = layers.Input(shape=(32, 32, 3))

    # Apply average pooling layers with different pooling windows and strides
    pool1 = layers.AveragePooling2D(pool_size=1, strides=1)(inputs)
    pool2 = layers.AveragePooling2D(pool_size=2, strides=2)(inputs)
    pool3 = layers.AveragePooling2D(pool_size=4, strides=4)(inputs)

    # Flatten the outputs of the pooling layers
    flatten1 = layers.Flatten()(pool1)
    flatten2 = layers.Flatten()(pool2)
    flatten3 = layers.Flatten()(pool3)

    # Concatenate the flattened features
    concat = layers.concatenate([flatten1, flatten2, flatten3])

    # Add two fully connected layers
    fc1 = layers.Dense(128, activation='relu')(concat)
    outputs = layers.Dense(10, activation='softmax')(fc1)

    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model