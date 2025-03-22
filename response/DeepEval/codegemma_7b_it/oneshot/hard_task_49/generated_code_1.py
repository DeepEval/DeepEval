import keras
from keras.layers import Input, AveragePooling2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Conv2D

def depthwise_conv2d(input_tensor, kernel_size, strides=(1, 1), padding='same'):
    channel_axis = 3 if keras.backend.image_data_format() == 'channels_last' else 1
    return DepthwiseConv2D(kernel_size, strides=strides, padding=padding, use_bias=False)(input_tensor)

def pointwise_conv2d(input_tensor, filters, kernel_size=(1, 1), strides=(1, 1), padding='same', activation=None):
    return Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False, activation=activation)(input_tensor)

def block1(input_tensor):
    avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='same')(input_tensor)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=1, padding='same')(input_tensor)
    avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=1, padding='same')(input_tensor)

    avg_pool1 = Flatten()(avg_pool1)
    avg_pool2 = Flatten()(avg_pool2)
    avg_pool3 = Flatten()(avg_pool3)

    concat_vector = Concatenate()([avg_pool1, avg_pool2, avg_pool3])

    return concat_vector

def block2(input_tensor):
    input_tensor = Lambda(lambda x: tf.split(x, 4, axis=3))(input_tensor)

    conv1 = Lambda(depthwise_conv2d, arguments={'kernel_size': (1, 1)})([input_tensor[0]])
    conv1 = Lambda(pointwise_conv2d, arguments={'filters': 16, 'kernel_size': (1, 1), 'activation': 'relu'})(conv1)

    conv2 = Lambda(depthwise_conv2d, arguments={'kernel_size': (3, 3)})([input_tensor[1]])
    conv2 = Lambda(pointwise_conv2d, arguments={'filters': 16, 'kernel_size': (1, 1), 'activation': 'relu'})(conv2)

    conv3 = Lambda(depthwise_conv2d, arguments={'kernel_size': (5, 5)})([input_tensor[2]])
    conv3 = Lambda(pointwise_conv2d, arguments={'filters': 16, 'kernel_size': (1, 1), 'activation': 'relu'})(conv3)

    conv4 = Lambda(depthwise_conv2d, arguments={'kernel_size': (7, 7)})([input_tensor[3]])
    conv4 = Lambda(pointwise_conv2d, arguments={'filters': 16, 'kernel_size': (1, 1), 'activation': 'relu'})(conv4)

    concat_vector = Concatenate()([conv1, conv2, conv3, conv4])

    return concat_vector

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    reshape_layer = Reshape((-1, 1, 1))(input_layer)

    block1_output = block1(input_tensor=reshape_layer)

    block2_output = block2(input_tensor=block1_output)

    flatten_layer = Flatten()(block2_output)

    dense = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=dense)

    return model