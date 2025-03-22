import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, concatenate, BatchNormalization, Flatten, Dense, Dropout, Add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    conv1_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1x1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1_3x3)

    conv2_1x1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pool1)
    conv2_3x3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_1x1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2_3x3)

    conv3_1x1 = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pool2)
    conv3_3x3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3_1x1)
    conv3_5x5 = Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv3_1x1)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv3_5x5)

    # Branch path
    conv_shortcut = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pool1)
    branch_conv1_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pool1)
    branch_conv1_3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_conv1_1x1)

    branch_conv2_avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(pool1)
    branch_conv2_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_conv2_avg_pool)
    branch_conv2_3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_conv2_1x1)

    branch_conv3_avg_pool = AveragePooling2D(pool_size=(4, 4), strides=4, padding='valid')(pool1)
    branch_conv3_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_conv3_avg_pool)
    branch_conv3_3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_conv3_1x1)

    branch_upsample = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='valid')(branch_conv3_3x3)
    branch_concat = concatenate([branch_upsample, branch_conv2_3x3], axis=3)
    branch_conv_concat = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_concat)

    branch_upsample_2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='valid')(branch_conv_concat)
    branch_concat_2 = concatenate([branch_upsample_2, branch_conv1_3x3], axis=3)
    branch_conv_concat_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_concat_2)

    # Main path and branch path fusion
    concat = concatenate([branch_conv_concat_2, conv3_5x5], axis=3)
    conv_1x1 = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)
    conv_drop = Dropout(rate=0.5)(conv_1x1)

    # Classification layer
    flatten_layer = Flatten()(conv_drop)
    dense1 = Dense(units=512, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model