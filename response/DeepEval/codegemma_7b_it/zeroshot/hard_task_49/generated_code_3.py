from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(28, 28, 1))

    # First block
    x = inputs
    x = layers.Rescaling(1./255)(x)
    x = layers.AveragePooling2D(pool_size=1, strides=1)(x)
    x = layers.AveragePooling2D(pool_size=2, strides=2)(x)
    x = layers.AveragePooling2D(pool_size=4, strides=4)(x)
    x = layers.Flatten()(x)
    concat_1 = x

    # Second block
    x = layers.Dense(4)(concat_1)
    x = layers.Reshape((2, 2, 2))(x)
    group_1 = tf.split(x, 4, axis=-1)
    group_1 = layers.Lambda(lambda x: x)(group_1[0])
    group_2 = layers.Lambda(lambda x: x)(group_1[1])
    group_3 = layers.Lambda(lambda x: x)(group_1[2])
    group_4 = layers.Lambda(lambda x: x)(group_1[3])
    group_1 = layers.DepthwiseSeparableConv2D(kernel_size=1, padding='same', use_bias=False)(group_1)
    group_1 = layers.BatchNormalization()(group_1)
    group_2 = layers.DepthwiseSeparableConv2D(kernel_size=3, padding='same', use_bias=False)(group_2)
    group_2 = layers.BatchNormalization()(group_2)
    group_3 = layers.DepthwiseSeparableConv2D(kernel_size=5, padding='same', use_bias=False)(group_3)
    group_3 = layers.BatchNormalization()(group_3)
    group_4 = layers.DepthwiseSeparableConv2D(kernel_size=7, padding='same', use_bias=False)(group_4)
    group_4 = layers.BatchNormalization()(group_4)
    concat_2 = layers.concatenate([group_1, group_2, group_3, group_4])

    # Final block
    x = layers.Flatten()(concat_2)
    x = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=x, name='mnist_model')
    return model