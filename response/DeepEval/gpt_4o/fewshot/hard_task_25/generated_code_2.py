import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Conv2DTranspose, Concatenate, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path: initial 1x1 convolution
    main_conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # First branch: 3x3 convolution
    branch1_conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_conv1x1)

    # Second branch: Average pooling -> 3x3 convolution -> Transpose convolution
    branch2_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(main_conv1x1)
    branch2_conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2_pool)
    branch2_transpose_conv = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')(branch2_conv3x3)

    # Third branch: Average pooling -> 3x3 convolution -> Transpose convolution
    branch3_pool = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(main_conv1x1)
    branch3_conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3_pool)
    branch3_transpose_conv = Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(4, 4), padding='same', activation='relu')(branch3_conv3x3)

    # Concatenating the outputs of all branches
    concat_branches = Concatenate()([branch1_conv3x3, branch2_transpose_conv, branch3_transpose_conv])

    # Final 1x1 convolution in the main path
    main_path_output = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_branches)

    # Branch path: 1x1 convolution to match the number of channels
    branch_path = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Fuse main path and branch path through addition
    fused_output = Add()([main_path_output, branch_path])

    # Flatten and fully connected layer for classification
    flatten_layer = Flatten()(fused_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model