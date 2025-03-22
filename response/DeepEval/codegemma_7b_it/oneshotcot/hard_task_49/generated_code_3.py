import keras
from keras.layers import Input, AveragePooling2D, Flatten, Dense, Lambda, Conv2D, Concatenate

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # First Block
    avg_pool_1x1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='same')(input_layer)
    avg_pool_2x2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    avg_pool_4x4 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='same')(input_layer)

    # Flatten and Concatenate
    flatten_1x1 = Flatten()(avg_pool_1x1)
    flatten_2x2 = Flatten()(avg_pool_2x2)
    flatten_4x4 = Flatten()(avg_pool_4x4)
    concat = Concatenate()([flatten_1x1, flatten_2x2, flatten_4x4])

    # Fully Connected Layer and Reshape
    dense_block = Dense(units=128, activation='relu')(concat)
    reshape = Lambda(lambda x: tf.reshape(x, (-1, 4, 4, 32)))(dense_block)

    # Second Block
    conv_1x1 = Lambda(lambda x: tf.split(x, 4, axis=-1))(reshape)
    conv_3x3 = Lambda(lambda x: tf.split(x, 4, axis=-1))(reshape)
    conv_5x5 = Lambda(lambda x: tf.split(x, 4, axis=-1))(reshape)
    conv_7x7 = Lambda(lambda x: tf.split(x, 4, axis=-1))(reshape)

    conv_1x1_output = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_1x1)
    conv_3x3_output = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_3x3)
    conv_5x5_output = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv_5x5)
    conv_7x7_output = Conv2D(filters=32, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(conv_7x7)

    # Concatenate and Flatten
    concat_output = Concatenate()([conv_1x1_output, conv_3x3_output, conv_5x5_output, conv_7x7_output])
    flatten_output = Flatten()(concat_output)

    # Output Layer
    output_layer = Dense(units=10, activation='softmax')(flatten_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model