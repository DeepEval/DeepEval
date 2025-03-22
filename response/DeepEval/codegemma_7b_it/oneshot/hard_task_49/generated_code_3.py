import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Reshape
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.regularizers import l2

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # First block
    avg_pool_1x1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_layer)
    avg_pool_2x2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    avg_pool_4x4 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(input_layer)

    flatten_1x1 = Flatten()(avg_pool_1x1)
    flatten_2x2 = Flatten()(avg_pool_2x2)
    flatten_4x4 = Flatten()(avg_pool_4x4)

    concat_block_1 = Concatenate()([flatten_1x1, flatten_2x2, flatten_4x4])

    # Fully connected layer and reshape
    fc1 = Dense(units=128, activation='relu')(concat_block_1)
    reshape_layer = Reshape((4, 4, 1))(fc1)

    # Second block
    def block(input_tensor):
        depthwise_conv_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same',
                                  activation='relu', use_bias=False, kernel_initializer=he_normal,
                                  kernel_regularizer=l2(0.0001))(input_tensor)
        depthwise_conv_3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                  activation='relu', use_bias=False, kernel_initializer=he_normal,
                                  kernel_regularizer=l2(0.0001))(input_tensor)
        depthwise_conv_5x5 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same',
                                  activation='relu', use_bias=False, kernel_initializer=he_normal,
                                  kernel_regularizer=l2(0.0001))(input_tensor)
        depthwise_conv_7x7 = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same',
                                  activation='relu', use_bias=False, kernel_initializer=he_normal,
                                  kernel_regularizer=l2(0.0001))(input_tensor)
        output_tensor = Concatenate()([depthwise_conv_1x1, depthwise_conv_3x3, depthwise_conv_5x5, depthwise_conv_7x7])

        return output_tensor

    block_output = block(reshape_layer)

    # Flatten and fully connected layer
    flatten_block_2 = Flatten()(block_output)
    fc2 = Dense(units=10, activation='softmax')(flatten_block_2)

    model = keras.Model(inputs=input_layer, outputs=fc2)

    return model