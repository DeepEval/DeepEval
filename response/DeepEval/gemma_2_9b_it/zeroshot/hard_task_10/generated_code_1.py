import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    input_img = layers.Input(shape=(32, 32, 3))

    # Path 1: 1x1 convolution
    x1 = layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_img)

    # Path 2: Sequence of convolutions
    x2 = layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_img)
    x2 = layers.Conv2D(filters=64, kernel_size=(1, 7), activation='relu')(x2)
    x2 = layers.Conv2D(filters=64, kernel_size=(7, 1), activation='relu')(x2)

    # Concatenate outputs from both paths
    x = layers.concatenate([x1, x2], axis=-1)
    x = layers.Conv2D(filters=3, kernel_size=(1, 1), activation='relu')(x)

    # Direct connection branch
    x_branch = layers.Conv2D(filters=3, kernel_size=(1, 1), activation='relu')(input_img)

    # Merge outputs through addition
    x = layers.add([x, x_branch])

    # Classification layers
    x = layers.Flatten()(x)
    x = layers.Dense(units=10, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_img, outputs=x)
    return model