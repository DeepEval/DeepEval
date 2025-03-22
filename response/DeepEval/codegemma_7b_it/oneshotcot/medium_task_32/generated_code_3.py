import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def depthwise_separable_conv(input_tensor, kernel_size, filters, strides):
        depthwise = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', depthwise_initializer='he_normal', use_bias=False)(input_tensor)
        pointwise = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', use_bias=False)(depthwise)
        return pointwise

    group1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    group1 = Lambda(lambda x: tf.map_fn(depthwise_separable_conv, x, [ (1, 1), 32, (1, 1)]) )(group1)

    group2 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    group2 = Lambda(lambda x: tf.map_fn(depthwise_separable_conv, x, [ (1, 1), 64, (1, 1)]) )(group2)

    group3 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    group3 = Lambda(lambda x: tf.map_fn(depthwise_separable_conv, x, [ (1, 1), 128, (1, 1)]) )(group3)

    concat = Concatenate(axis=-1)([group1, group2, group3])
    bn = BatchNormalization()(concat)
    flatten = Flatten()(bn)
    dense = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=dense)

    return model