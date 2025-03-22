import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():

    # Input layer
    input_img = layers.Input(shape=(32, 32, 3), name='image_input')

    # First block
    x1 = layers.Lambda(tf.split)(input_img, num_or_size_splits=3, axis=3)
    x1 = [layers.Conv2D(filters=32, kernel_size=(k, k), padding='same', use_bias=False)(xi) for k, xi in enumerate([x1[0], x1[1], x1[2]], start=1)]
    x1 = layers.Concatenate(axis=3)(x1)

    # Second block
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x1)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x2 = layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same')(x)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Activation('relu')(x2)
    x2 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Activation('relu')(x2)
    x2 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Activation('relu')(x2)

    x3 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = layers.Concatenate(axis=3)([x2, x3])

    # Output layer
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(10, activation='softmax', name='image_output')(x)

    # Model construction
    model = models.Model(inputs=input_img, outputs=x, name='custom_model')

    return model