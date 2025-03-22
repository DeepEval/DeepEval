import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Reshape

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    max_pooling_1x1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_layer)
    max_pooling_2x2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    max_pooling_4x4 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(input_layer)

    flatten_1x1 = Flatten()(max_pooling_1x1)
    flatten_2x2 = Flatten()(max_pooling_2x2)
    flatten_4x4 = Flatten()(max_pooling_4x4)

    concat_1 = Concatenate()([flatten_1x1, flatten_2x2, flatten_4x4])
    drop_out_1 = Dropout(0.5)(concat_1)

    # Block 2
    conv_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(drop_out_1)
    reshaped_conv_1x1 = Reshape((conv_1x1.shape[1], conv_1x1.shape[2], 64))(conv_1x1)

    conv_1x7_7x1 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(reshaped_conv_1x1)
    reshaped_conv_1x7_7x1 = Reshape((conv_1x7_7x1.shape[1], conv_1x7_7x1.shape[2], 64))(conv_1x7_7x1)
    conv_7x1_1x7 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(reshaped_conv_1x7_7x1)

    conv_1x7_1x7 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(reshaped_conv_1x1)
    reshaped_conv_1x7_1x7 = Reshape((conv_1x7_1x7.shape[1], conv_1x7_1x7.shape[2], 64))(conv_1x7_1x7)
    conv_7x1_1x7 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(reshaped_conv_1x7_1x7)

    conv_avg_pool = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(drop_out_1)
    reshaped_conv_avg_pool = Reshape((conv_avg_pool.shape[1], conv_avg_pool.shape[2], 64))(conv_avg_pool)
    avg_pool = MaxPooling2D(pool_size=(conv_avg_pool.shape[1], conv_avg_pool.shape[2]), strides=1, padding='valid')(reshaped_conv_avg_pool)

    concat_2 = Concatenate()([conv_1x1, conv_1x7_7x1, conv_7x1_1x7, avg_pool])

    # Block 3
    flatten_2 = Flatten()(concat_2)
    dense_1 = Dense(units=256, activation='relu')(flatten_2)
    drop_out_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(units=10, activation='softmax')(drop_out_2)

    model = keras.Model(inputs=input_layer, outputs=dense_2)

    return model