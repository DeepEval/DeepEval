import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def depthwise_separable_conv2d(input_tensor, filters, kernel_size, strides=(1, 1), padding='valid'):
    # Depthwise convolution
    depthwise_conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False)(input_tensor)
    depthwise_conv = BatchNormalization()(depthwise_conv)
    depthwise_conv = Activation('relu')(depthwise_conv)

    # Pointwise convolution
    pointwise_conv = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)(depthwise_conv)
    pointwise_conv = BatchNormalization()(pointwise_conv)
    pointwise_conv = Activation('relu')(pointwise_conv)

    return pointwise_conv

def block(input_tensor):
    path1 = depthwise_separable_conv2d(input_tensor, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')
    path1 = Dropout(rate=0.2)(path1)
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)(path1)
    path1 = BatchNormalization()(path1)
    path1 = Activation('relu')(path1)

    path2 = depthwise_separable_conv2d(input_tensor, filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same')
    path2 = Dropout(rate=0.2)(path2)
    path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)(path2)
    path2 = BatchNormalization()(path2)
    path2 = Activation('relu')(path2)

    path3 = depthwise_separable_conv2d(input_tensor, filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same')
    path3 = Dropout(rate=0.2)(path3)
    path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)(path3)
    path3 = BatchNormalization()(path3)
    path3 = Activation('relu')(path3)

    path4 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
    path4 = Dropout(rate=0.2)(path4)
    path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)(path4)
    path4 = BatchNormalization()(path4)
    path4 = Activation('relu')(path4)

    output_tensor = Concatenate()([path1, path2, path3, path4])

    return output_tensor

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)

    block_output = block(input_tensor=max_pooling)
    bath_norm = BatchNormalization()(block_output)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model