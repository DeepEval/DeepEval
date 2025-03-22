import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Reshape, Permute

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)

    # Block 1
    split_input = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))(conv)
    group1 = Lambda(lambda x: tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x))(split_input[0])
    group1 = Lambda(lambda x: tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x))(group1)
    group1 = Lambda(lambda x: tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x))(group1)
    group2 = Lambda(lambda x: tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x))(split_input[1])
    concat = Concatenate(axis=-1)([group1, group2])

    # Block 2
    input_shape = keras.backend.int_shape(concat)
    reshape = Reshape((input_shape[1], input_shape[2], input_shape[3] // 4, 4))(concat)
    permute = Permute((1, 2, 4, 3))(reshape)
    reshape_back = Reshape((input_shape[1], input_shape[2], input_shape[3]))(permute)

    flatten_layer = Flatten()(reshape_back)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model