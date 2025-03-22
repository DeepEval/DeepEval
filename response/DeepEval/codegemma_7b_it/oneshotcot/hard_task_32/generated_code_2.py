import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def depthwise_separable_conv2d(input_tensor, filters, kernel_size, strides=(1, 1), padding='valid'):
    conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False)(input_tensor)
    bn = BatchNormalization()(conv)
    act = keras.activations.relu(bn)
    return act

def pointwise_conv2d(input_tensor, filters, kernel_size=(1, 1), strides=(1, 1), padding='valid'):
    conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False)(input_tensor)
    bn = BatchNormalization()(conv)
    act = keras.activations.relu(bn)
    return act

def block(input_tensor):
    branch1 = pointwise_conv2d(input_tensor, filters=32, kernel_size=(1, 1))
    branch1 = depthwise_separable_conv2d(branch1, filters=32, kernel_size=(3, 3))
    branch1 = pointwise_conv2d(branch1, filters=32, kernel_size=(1, 1))
    branch1 = Dropout(0.2)(branch1)

    branch2 = pointwise_conv2d(input_tensor, filters=64, kernel_size=(1, 1))
    branch2 = depthwise_separable_conv2d(branch2, filters=64, kernel_size=(5, 5))
    branch2 = pointwise_conv2d(branch2, filters=64, kernel_size=(1, 1))
    branch2 = Dropout(0.2)(branch2)

    branch3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_tensor)
    branch3 = pointwise_conv2d(branch3, filters=64, kernel_size=(1, 1))
    branch3 = Dropout(0.2)(branch3)

    concat = Concatenate()([branch1, branch2, branch3])
    return concat

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)

    branch1_output = block(input_tensor=max_pooling)
    branch2_output = block(input_tensor=max_pooling)
    branch3_output = block(input_tensor=max_pooling)

    concat = Concatenate()([branch1_output, branch2_output, branch3_output])

    flatten_layer = Flatten()(concat)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model