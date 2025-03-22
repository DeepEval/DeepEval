import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def depthwise_separable_conv2d(input_tensor, filters, kernel_size, strides=(1, 1), padding='same'):
    depthwise_conv = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False, depth_wise_initializer='he_normal')(input_tensor)
    pointwise_conv = Conv2D(filters, (1, 1), strides=1, padding='same', use_bias=False, kernel_initializer='he_normal')(depthwise_conv)
    return pointwise_conv

def block(input_tensor):
    path1 = depthwise_separable_conv2d(input_tensor, 32, (3, 3), padding='same', strides=(1, 1))
    path1 = Dropout(0.2)(path1)
    path1 = Conv2D(32, (1, 1), padding='same', strides=(1, 1), use_bias=False)(path1)
    path1 = BatchNormalization()(path1)

    path2 = depthwise_separable_conv2d(input_tensor, 32, (5, 5), padding='same', strides=(1, 1))
    path2 = Dropout(0.2)(path2)
    path2 = Conv2D(32, (1, 1), padding='same', strides=(1, 1), use_bias=False)(path2)
    path2 = BatchNormalization()(path2)

    path3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
    path3 = Conv2D(32, (1, 1), padding='same', strides=(1, 1), use_bias=False)(path3)
    path3 = BatchNormalization()(path3)

    output_tensor = Concatenate()([path1, path2, path3])
    return output_tensor

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)

    branch1 = block(input_tensor=max_pooling)
    branch2 = block(input_tensor=max_pooling)
    branch3 = block(input_tensor=max_pooling)

    concat = Concatenate()([branch1, branch2, branch3])
    flatten_layer = Flatten()(concat)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model