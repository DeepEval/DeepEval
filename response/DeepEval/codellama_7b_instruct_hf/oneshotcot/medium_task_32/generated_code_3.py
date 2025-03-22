import keras
from keras.layers import Input, Lambda, Flatten, Dense
from keras.applications.vgg16 import VGG16


def dl_model():
    input_shape = (32, 32, 3)

    split_layer = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_shape)

    depthwise_conv1 = VGG16(include_top=False, input_shape=split_layer.shape[1:], depthwise_conv_layer=True)(split_layer)
    depthwise_conv2 = VGG16(include_top=False, input_shape=split_layer.shape[1:], depthwise_conv_layer=True)(split_layer)
    depthwise_conv3 = VGG16(include_top=False, input_shape=split_layer.shape[1:], depthwise_conv_layer=True)(split_layer)

    concatenated_layer = Concatenate()([depthwise_conv1, depthwise_conv2, depthwise_conv3])

    flatten_layer = Flatten()(concatenated_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_shape, outputs=output_layer)

    return model