import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Concatenate, Dense, Lambda, Reshape, tf

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # First block
    avg_pool_1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    avg_pool_2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    avg_pool_3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)
    
    flat_1 = Flatten()(avg_pool_1)
    flat_2 = Flatten()(avg_pool_2)
    flat_3 = Flatten()(avg_pool_3)
    
    concat_layer = Concatenate()([flat_1, flat_2, flat_3])

    dense_layer = Dense(units=128, activation='relu')(concat_layer)
    reshape_layer = Reshape((1, 128))(dense_layer)

    # Second block
    split_layer = Lambda(lambda x: tf.split(x, 4, axis=1))(reshape_layer)

    # Depthwise separable convolutions
    conv_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)(split_layer[0])
    conv_3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(split_layer[1])
    conv_5x5 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False)(split_layer[2])
    conv_7x7 = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same', use_bias=False)(split_layer[3])

    # Concatenate outputs
    concat_output = Concatenate(axis=1)([conv_1x1, conv_3x3, conv_5x5, conv_7x7])
    
    # Final layers
    flat_output = Flatten()(concat_output)
    output_layer = Dense(units=10, activation='softmax')(flat_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model