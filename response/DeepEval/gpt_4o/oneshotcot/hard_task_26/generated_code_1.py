import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Add, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Initial 1x1 convolutional layer in the main path
    main_conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 1: 3x3 convolutional layer
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_conv1x1)

    # Branch 2: MaxPooling -> 3x3 Convolution -> Upsampling
    branch2_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(main_conv1x1)
    branch2_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2_pool)
    branch2_upsample = UpSampling2D(size=(2, 2))(branch2_conv)

    # Branch 3: MaxPooling -> 3x3 Convolution -> Upsampling
    branch3_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(main_conv1x1)
    branch3_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3_pool)
    branch3_upsample = UpSampling2D(size=(2, 2))(branch3_conv)

    # Concatenate outputs from the branches
    concat_branches = Concatenate()([branch1, branch2_upsample, branch3_upsample])

    # Final 1x1 convolutional layer in the main path
    final_main_conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_branches)

    # Branch path: Initial 1x1 convolutional layer
    branch_path_conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Add outputs of main and branch paths
    combined_output = Add()([final_main_conv1x1, branch_path_conv1x1])

    # Fully connected layers for classification
    flatten_layer = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model