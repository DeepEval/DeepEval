from tensorflow import keras
from tensorflow.keras import layers

def depthwise_conv_block(x, filters, kernel_size, padding='same', alpha=1.0):
    channel_axis = 1 if keras.backend.image_data_format() == 'channels_first' else -1
    x = layers.ZeroPadding2D(padding=padding)(x)
    x = layers.DepthwiseConv2D(kernel_size, strides=(1, 1), padding='valid', depthwise_initializer='he_normal', use_bias=False)(x)
    x = layers.BatchNormalization(axis=channel_axis)(x)
    x = layers.LeakyReLU(alpha=alpha)(x)
    x = layers.ZeroPadding2D(padding=padding)(x)
    x = layers.Conv2D(filters, (1, 1), strides=(1, 1), padding='valid', kernel_initializer='he_normal', use_bias=False)(x)
    x = layers.BatchNormalization(axis=channel_axis)(x)
    x = layers.LeakyReLU(alpha=alpha)(x)
    return x

def residual_conv_block(x, filters):
    conv = depthwise_conv_block(x, filters, kernel_size=(3, 3))
    conv = layers.Dropout(rate=0.2)(conv)
    shortcut = layers.Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
    shortcut = layers.BatchNormalization(axis=1)(shortcut)
    res_path = layers.add([shortcut, conv])
    res_path = layers.LeakyReLU()(res_path)
    return res_path

def classifier(input_shape):
    input_img = keras.Input(shape=input_shape)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_img)
    x = layers.Dropout(rate=0.2)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x_1 = residual_conv_block(x, 64)

    x_2 = residual_conv_block(x, 128)
    x_2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x_2)

    x_3 = residual_conv_block(x, 256)
    x_3 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x_3)

    concat = layers.concatenate([x_1, x_2, x_3], axis=3)

    concat = layers.BatchNormalization()(concat)
    concat = layers.LeakyReLU()(concat)
    concat = layers.Dropout(rate=0.5)(concat)

    flatten = layers.Flatten()(concat)
    output = layers.Dense(10, activation='softmax')(flatten)
    model = keras.Model(inputs=input_img, outputs=output)
    return model

def dl_model():
    input_shape = (28, 28, 1)
    model = classifier(input_shape)
    return model

model = dl_model()