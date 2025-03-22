import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add, Conv2DTranspose, AveragePooling2D

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    conv_main = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_main = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_main)
    pool_main = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_main)

    # Branch path
    conv_branch = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_branch = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_branch)

    # Average pooling and convolutional layers for downsampling
    conv_branch_avg = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_branch)
    conv_branch_avg = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_branch_avg)
    conv_branch_avg = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_branch_avg)

    # Average pooling and convolutional layers for downsampling
    conv_branch_max = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_branch)
    conv_branch_max = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_branch_max)
    conv_branch_max = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_branch_max)

    # Upsampling and convolutional layers for upsampling
    conv_branch_upsample = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu')(conv_branch)

    # Concatenate outputs
    concat_layers = Concatenate()([pool_main, conv_branch_upsample, conv_branch_max, conv_branch_avg])

    # Main path output
    conv_main_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_layers)
    pool_main_output = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_main_output)

    # Branch path output
    conv_branch_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_layers)

    # Fuse main path and branch path outputs
    concat_outputs = Add()([pool_main_output, conv_branch_output])

    # Fully connected layer
    flatten_layer = Flatten()(concat_outputs)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model