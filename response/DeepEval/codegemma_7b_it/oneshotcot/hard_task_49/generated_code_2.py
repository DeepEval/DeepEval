import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D, Reshape, Lambda, DepthwiseConv2D

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # First block
    avg_pool_1x1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_layer)
    avg_pool_2x2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    avg_pool_4x4 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='valid')(input_layer)

    avg_pool_1x1_flat = Flatten()(avg_pool_1x1)
    avg_pool_2x2_flat = Flatten()(avg_pool_2x2)
    avg_pool_4x4_flat = Flatten()(avg_pool_4x4)

    avg_pool_concat = Concatenate()([avg_pool_1x1_flat, avg_pool_2x2_flat, avg_pool_4x4_flat])

    # Reshape
    reshape_layer = Reshape((3, 56))(avg_pool_concat)

    # Second block
    split_layer = Lambda(lambda x: tf.split(x, 4, axis=-1))(reshape_layer)

    depthwise_conv_1x1 = Lambda(lambda x: DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(x))(split_layer[0])
    depthwise_conv_3x3 = Lambda(lambda x: DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(x))(split_layer[1])
    depthwise_conv_5x5 = Lambda(lambda x: DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(x))(split_layer[2])
    depthwise_conv_7x7 = Lambda(lambda x: DepthwiseConv2D(kernel_size=(7, 7), padding='same', activation='relu')(x))(split_layer[3])

    concat = Concatenate()([depthwise_conv_1x1, depthwise_conv_3x3, depthwise_conv_5x5, depthwise_conv_7x7])

    # Flatten
    flatten_layer = Flatten()(concat)

    # Fully connected layer
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Model
    model = keras.Model(inputs=input_layer, outputs=dense_layer)

    return model