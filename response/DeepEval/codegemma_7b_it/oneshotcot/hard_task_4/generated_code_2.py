import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Conv2DTranspose, Lambda, Multiply

def channel_attention(input_tensor, ratio):
    avg_pool = GlobalAveragePooling2D()(input_tensor)
    fc1 = Dense(units=int(input_tensor.shape[-1] // ratio), activation='relu')(avg_pool)
    fc2 = Dense(units=input_tensor.shape[-1], activation='sigmoid')(fc1)
    return Multiply()([fc2, input_tensor])

def depthwise_conv(input_tensor, filters, kernel_size, strides, padding):
    return DepthwiseConv2D(kernel_size, strides=strides, padding=padding, use_bias=False)(input_tensor)

def pointwise_conv(input_tensor, filters, strides, padding):
    return Conv2D(filters, kernel_size=(1, 1), strides=strides, padding=padding, use_bias=False)(input_tensor)

def residual_block(input_tensor, filters, kernel_size, strides, padding):
    conv1 = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False)(input_tensor)
    bn1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(bn1)

    conv2 = Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding=padding, use_bias=False)(act1)
    bn2 = BatchNormalization()(conv2)

    if strides != 1 or input_tensor.shape[-1] != filters:
        shortcut = Conv2D(filters=filters, kernel_size=(1, 1), strides=strides, padding=padding, use_bias=False)(input_tensor)
        shortcut = BatchNormalization()(shortcut)
    else:
        shortcut = input_tensor

    output = Add()([shortcut, bn2])
    act2 = Activation('relu')(output)

    return act2

def make_dense_block(input_tensor, filters, kernel_size):
    dense_block = input_tensor
    for i in range(4):
        dense_block = concatenate([dense_block, residual_block(dense_block, filters, kernel_size, strides=1, padding='same')])
    return dense_block

def make_dense_net(input_tensor, filters, growth_rate, num_classes):
    dense_net = input_tensor
    dense_net = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same')(dense_net)
    dense_net = BatchNormalization()(dense_net)
    dense_net = Activation('relu')(dense_net)

    dense_net = make_dense_block(dense_net, filters, kernel_size=3)
    dense_net = BatchNormalization()(dense_net)
    dense_net = Activation('relu')(dense_net)

    dense_net = Conv2D(filters=num_classes, kernel_size=(1, 1), strides=1, padding='same')(dense_net)
    dense_net = BatchNormalization()(dense_net)

    dense_net = Flatten()(dense_net)
    dense_net = Dense(units=num_classes, activation='softmax')(dense_net)

    return dense_net

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    conv_input = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(input_layer)
    conv_input = BatchNormalization()(conv_input)
    conv_input = Activation('relu')(conv_input)

    depthwise_conv_input = depthwise_conv(conv_input, filters=64, kernel_size=(3, 3), strides=1, padding='same')
    pointwise_conv_input = pointwise_conv(depthwise_conv_input, filters=256, strides=1, padding='same')

    channel_attention_input = channel_attention(pointwise_conv_input, ratio=4)

    conv_output = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same')(channel_attention_input)

    output_layer = make_dense_net(input_tensor=conv_output, filters=64, growth_rate=32, num_classes=10)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model