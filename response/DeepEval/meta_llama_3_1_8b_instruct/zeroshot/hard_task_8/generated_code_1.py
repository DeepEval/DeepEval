import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Input layer
    inputs = keras.Input(shape=(28, 28, 1))

    # Block 1
    x = layers.Conv2D(6, 1, activation='relu')(inputs)
    x = layers DepthwiseConv2D(3, activation='relu')(x)
    x = layers.Conv2D(6, 1, activation='relu')(x)

    x_branch = layers.DepthwiseConv2D(3, activation='relu')(x)
    x_branch = layers.Conv2D(6, 1, activation='relu')(x_branch)

    x = layers.Concatenate()([x, x_branch])

    # Block 2
    x = layers.Reshape((-1,))(x)
    x = layers.Lambda(lambda x: tf.reshape(x, [-1, 7, 7, 12, 6]))(x)
    x = layers.Permute((1, 2, 4, 0, 3))(x)
    x = layers.Reshape((-1, 6, 7, 7))(x)

    # Classification block
    x = layers.Flatten()(x)
    x = layers.Dense(10, activation='softmax')(x)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=x)

    return model

# Test the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])