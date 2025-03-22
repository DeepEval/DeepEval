import keras
from keras.layers import Input, Lambda, Flatten, Dense


def dl_model():
    input_layer = Input(shape=(32, 32, 3))


    group_1, group_2, group_3 = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)


    conv_1 = Lambda(lambda x: tf.depthwise_separable_conv2d(x, [3, 3, 1, 1], [1, 1, 1, 1], 'VALID'))(group_1)
    conv_2 = Lambda(lambda x: tf.depthwise_separable_conv2d(x, [3, 3, 2, 2], [1, 1, 1, 1], 'VALID'))(group_2)
    conv_3 = Lambda(lambda x: tf.depthwise_separable_conv2d(x, [3, 3, 3, 3], [1, 1, 1, 1], 'VALID'))(group_3)


    concat = Lambda(lambda x: tf.concat(x, axis=1))([conv_1, conv_2, conv_3])


    flatten = Flatten()(concat)


    output = Dense(units=10, activation='softmax')(flatten)


    model = keras.Model(inputs=input_layer, outputs=output)


    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    return model