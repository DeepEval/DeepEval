import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, UpSampling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    conv_1x1_main = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_3x3_main = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_1x1_main)

    # Branch path
    conv_1x1_branch = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Downsample and process in the second branch
    avg_pool_branch_1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_1x1_branch)
    conv_3x3_branch_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(avg_pool_branch_1)

    # Downsample and process in the third branch
    avg_pool_branch_2 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='valid')(conv_1x1_branch)
    conv_3x3_branch_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(avg_pool_branch_2)

    # Upsample and concatenate from both branches
    up_sample_branch_1 = UpSampling2D(size=(2, 2))(conv_3x3_branch_1)
    up_sample_branch_2 = UpSampling2D(size=(4, 4))(conv_3x3_branch_2)
    concat_branches = Concatenate()([up_sample_branch_1, up_sample_branch_2, conv_3x3_main])

    # Main path output
    conv_1x1_main_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_branches)

    # Branch path output
    conv_1x1_branch_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_1x1_branch)

    # Fuse outputs and classify
    concat_outputs = Concatenate()([conv_1x1_main_output, conv_1x1_branch_output])
    flatten_layer = Flatten()(concat_outputs)
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Model
    model = keras.Model(inputs=input_layer, outputs=dense_layer)

    return model