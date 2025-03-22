import keras
from keras.layers import Input, Lambda, Concatenate, Dense, BatchNormalization, Flatten, SeparableConv2D
from keras.models import Model

def dl_model():
    # Block 1
    input_layer = Input(shape=(32, 32, 3))
    x = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    x1 = SeparableConv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(x[0])
    x2 = SeparableConv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(x[1])
    x3 = SeparableConv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same')(x[2])
    x = Concatenate()([x1, x2, x3])
    x = BatchNormalization()(x)

    # Block 2
    x = SeparableConv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)

    x1 = SeparableConv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x2 = SeparableConv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = Concatenate()([x1, x2])
    x = BatchNormalization()(x)

    x1 = SeparableConv2D(filters=16, kernel_size=(1, 3), strides=(1, 1), padding='same')(x)
    x2 = SeparableConv2D(filters=16, kernel_size=(3, 1), strides=(1, 1), padding='same')(x)
    x = Concatenate()([x1, x2])
    x = BatchNormalization()(x)

    x1 = SeparableConv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x2 = SeparableConv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = Concatenate()([x1, x2])
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(units=10)(x)
    x = BatchNormalization()(x)

    return Model(inputs=input_layer, outputs=x)