import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():

    # Input layer
    input_img = keras.Input(shape=(32, 32, 3))

    # First block
    x_1 = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(input_img)
    x_1 = [layers.SeparableConv2D(32, (1, 1), padding='same', activation='relu')(x) for x in x_1]
    x_1 = [layers.SeparableConv2D(64, (3, 3), padding='same', activation='relu')(x) for x in x_1]
    x_1 = [layers.SeparableConv2D(128, (5, 5), padding='same', activation='relu')(x) for x in x_1]
    x_1 = [layers.BatchNormalization()(x) for x in x_1]

    # Second block
    x_2 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(input_img)
    x_2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x_2)
    x_2 = layers.Conv2D(128, (5, 5), padding='same', activation='relu')(x_2)
    x_2 = layers.BatchNormalization()(x_2)

    x_3 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(input_img)
    x_3 = layers.Conv2D(64, (1, 7), padding='same', activation='relu')(x_3)
    x_3 = layers.Conv2D(128, (7, 1), padding='same', activation='relu')(x_3)
    x_3 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x_3)
    x_3 = layers.BatchNormalization()(x_3)

    x_4 = layers.AveragePooling2D((2, 2))(input_img)
    x_4 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x_4)
    x_4 = layers.BatchNormalization()(x_4)

    # Concatenate outputs from all branches
    merged = layers.concatenate([x_1[0], x_1[1], x_1[2], x_2, x_3, x_4])

    # Fully connected layers
    x = layers.GlobalAveragePooling2D()(merged)
    x = layers.Dense(512, activation='relu')(x)
    output = layers.Dense(10, activation='softmax')(x)

    # Create model
    model = keras.Model(inputs=input_img, outputs=output)

    return model