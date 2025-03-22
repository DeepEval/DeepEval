import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():

    inputs = keras.Input(shape=(32, 32, 3))

    # Block 1
    x1 = layers.Lambda(lambda x: tf.split(x, 3, axis=3))(inputs)
    x1 = [layers.SeparableConv2D(32, (1, 1), padding='same')(x) for x in x1]
    x1 = [layers.BatchNormalization(axis=3)(x) for x in x1]
    x1 = [layers.Activation('relu')(x) for x in x1]

    x2 = layers.Lambda(lambda x: tf.split(x, 3, axis=3))(inputs)
    x2 = [layers.SeparableConv2D(32, (3, 3), padding='same')(x) for x in x2]
    x2 = [layers.BatchNormalization(axis=3)(x) for x in x2]
    x2 = [layers.Activation('relu')(x) for x in x2]

    x3 = layers.Lambda(lambda x: tf.split(x, 3, axis=3))(inputs)
    x3 = [layers.SeparableConv2D(32, (5, 5), padding='same')(x) for x in x3]
    x3 = [layers.BatchNormalization(axis=3)(x) for x in x3]
    x3 = [layers.Activation('relu')(x) for x in x3]

    concat_x123 = layers.Concatenate(axis=3)([x1[0], x2[0], x3[0], x1[1], x2[1], x3[1], x1[2], x2[2], x3[2]])

    # Block 2
    path_1 = layers.Conv2D(32, (1, 1), padding='same')(concat_x123)
    path_2 = layers.AveragePooling2D((3, 3), padding='same')(concat_x123)
    path_2 = layers.Conv2D(32, (1, 1), padding='same')(path_2)
    path_3 = layers.Conv2D(32, (1, 1), padding='same')(concat_x123)
    path_3_1 = layers.Conv2D(32, (1, 3), padding='same')(path_3)
    path_3_2 = layers.Conv2D(32, (3, 1), padding='same')(path_3)
    concat_path_3 = layers.Concatenate(axis=3)([path_3_1, path_3_2])
    path_4 = layers.Conv2D(32, (1, 1), padding='same')(concat_x123)
    path_4 = layers.Conv2D(32, (3, 3), padding='same')(path_4)
    path_4_1 = layers.Conv2D(32, (1, 3), padding='same')(path_4)
    path_4_2 = layers.Conv2D(32, (3, 1), padding='same')(path_4)
    concat_path_4 = layers.Concatenate(axis=3)([path_4_1, path_4_2])

    concat_paths = layers.Concatenate(axis=3)([path_1, path_2, concat_path_3, concat_path_4])

    # Classification Layer
    flattened = layers.Flatten()(concat_paths)
    outputs = layers.Dense(10, activation='softmax')(flattened)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model