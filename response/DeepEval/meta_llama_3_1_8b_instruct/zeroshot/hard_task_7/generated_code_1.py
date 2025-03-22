import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the initial convolutional layer
    inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)

    # Define Block 1
    block1 = layers.Lambda(lambda x: tf.split(x, 2, axis=-1))(x)
    group1 = layers.Conv2D(32, (1, 1), activation='relu')(block1[0])
    group1 = layers.DepthwiseConv2D((3, 3), activation='relu')(group1)
    group1 = layers.Conv2D(32, (1, 1), activation='relu')(group1)
    group2 = block1[1]
    merged = layers.Concatenate()([group1, group2])

    # Define Block 2
    x = merged
    input_shape = tf.keras.Input(shape=(28, 28, 64)).shape
    height, width, channels = input_shape[1], input_shape[2], input_shape[3]
    channels_per_group = channels // 2
    x = layers.Lambda(lambda x: tf.reshape(x, (-1, height, width, 2, channels_per_group)))(x)
    x = layers.Permute((1, 2, 4, 0, 3))(x)
    x = layers.Reshape((height, width, channels))(x)

    # Define the fully connected layer
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = keras.Model(inputs=[inputs], outputs=outputs)
    return model

model = dl_model()
model.summary()