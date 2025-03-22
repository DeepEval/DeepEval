import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():

    # Input layer
    input_img = layers.Input(shape=(32, 32, 3))

    # Main path
    x1 = layers.Lambda(lambda x: tf.split(x, 3, axis=3))(input_img)
    x1 = [layers.SeparableConv2D(32, (1, 1), padding='same', activation='relu')(xi) for xi in x1]
    x1 = layers.Concatenate()(x1)

    # Branch path
    x2 = layers.SeparableConv2D(32, (1, 1), padding='same', activation='relu')(input_img)
    x2 = layers.SeparableConv2D(32, (3, 3), padding='same', activation='relu')(x2)
    x2 = layers.SeparableConv2D(32, (5, 5), padding='same', activation='relu')(x2)

    # Concatenate main and branch outputs
    merged = layers.Add()([x1, x2])

    # Fully connected layers
    x = layers.Flatten()(merged)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(10, activation='softmax')(x)

    # Model creation
    model = models.Model(inputs=input_img, outputs=x)

    return model