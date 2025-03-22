import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)
    
    def block_one(input_tensor):
        conv1_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv1_1x1_pool = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(conv1_1x1)
        conv1_1x1_pool_drop = Dropout(0.25)(conv1_1x1_pool)

        conv1_2x2 = Conv2D(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')(input_tensor)
        conv1_2x2_drop = Dropout(0.25)(conv1_2x2)

        conv1_4x4 = Conv2D(filters=64, kernel_size=(4, 4), strides=(4, 4), padding='same', activation='relu')(input_tensor)
        conv1_4x4_drop = Dropout(0.25)(conv1_4x4)

        concat_one = Concatenate()([conv1_1x1_pool_drop, conv1_2x2_drop, conv1_4x4_drop])
        return concat_one

    block_one_output = block_one(input_tensor=max_pooling)
    bath_norm = BatchNormalization()(block_one_output)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense1_drop = Dropout(0.5)(dense1)

    reshape_layer = keras.layers.Reshape((2,2,dense1.shape[1]))(dense1_drop)

    def block_two(input_tensor):
        conv2_1x1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)

        conv2_1x1_conv = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2_1x1)
        conv2_1x1_conv_drop = Dropout(0.25)(conv2_1x1_conv)

        conv2_3x3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2_3x3_drop = Dropout(0.25)(conv2_3x3)

        conv2_3x3_3x3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(3, 3), padding='same', activation='relu')(input_tensor)

        conv2_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        conv2_pool_conv = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2_pool)

        concat_two = Concatenate()([conv2_1x1, conv2_1x1_conv_drop, conv2_3x3, conv2_3x3_3x3, conv2_pool_conv])
        return concat_two

    block_two_output = block_two(input_tensor=reshape_layer)
    bath_norm_two = BatchNormalization()(block_two_output)
    flatten_layer_two = Flatten()(bath_norm_two)
    dense2 = Dense(units=100, activation='relu')(flatten_layer_two)
    dense2_drop = Dropout(0.5)(dense2)
    dense3 = Dense(units=64, activation='relu')(dense2_drop)
    dense3_drop = Dropout(0.5)(dense3)
    output_layer = Dense(units=10, activation='softmax')(dense3_drop)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model