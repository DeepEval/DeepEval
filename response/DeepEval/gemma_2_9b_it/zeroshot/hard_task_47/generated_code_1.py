import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    inputs = layers.Input(shape=(32, 32, 3))

    # First Block: Split and Depthwise Separable Convolutions
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(inputs)
    x = [
        layers.Conv2D(64, (1, 1), activation='relu', padding='same', name='conv1')(i)
        for i in x
    ]
    x = [
        layers.BatchNormalization()(i)
        for i in x
    ]
    x = [
        layers.DepthwiseConv2D(kernel_size=(3, 3), strides=1, activation='relu', padding='same', name='conv2')(i)
        for i in x
    ]
    x = [
        layers.BatchNormalization()(i)
        for i in x
    ]
    x = [
        layers.DepthwiseConv2D(kernel_size=(5, 5), strides=1, activation='relu', padding='same', name='conv3')(i)
        for i in x
    ]
    x = [
        layers.BatchNormalization()(i)
        for i in x
    ]
    x = layers.Concatenate(axis=-1)(x)

    # Second Block: Feature Extraction Branches
    branch1 = layers.Conv2D(128, (1, 1), activation='relu', padding='same')(x)
    branch1 = layers.BatchNormalization()(branch1)
    branch1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(branch1)
    branch1 = layers.BatchNormalization()(branch1)

    branch2 = layers.Conv2D(128, (1, 1), activation='relu', padding='same')(x)
    branch2 = layers.BatchNormalization()(branch2)
    branch2 = layers.Conv2D(128, (1, 7), activation='relu', padding='same')(branch2)
    branch2 = layers.BatchNormalization()(branch2)
    branch2 = layers.Conv2D(128, (7, 1), activation='relu', padding='same')(branch2)
    branch2 = layers.BatchNormalization()(branch2)
    branch2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(branch2)
    branch2 = layers.BatchNormalization()(branch2)

    branch3 = layers.AvgPool2D((3, 3), strides=2)(x)
    branch3 = layers.Conv2D(128, (1, 1), activation='relu', padding='same')(branch3)
    branch3 = layers.BatchNormalization()(branch3)

    # Concatenate branches
    x = layers.Concatenate(axis=-1)([branch1, branch2, branch3])

    # Fully Connected Layers
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model