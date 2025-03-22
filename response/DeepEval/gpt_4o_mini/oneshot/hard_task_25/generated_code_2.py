import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Conv2DTranspose, Concatenate, BatchNormalization, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 1: 3x3 convolution
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)

    # Branch 2: Average pooling followed by 3x3 conv, then upsample
    branch2_downsample = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(main_path)
    branch2_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2_downsample)
    branch2_upsample = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=2, padding='same')(branch2_conv)

    # Branch 3: Average pooling followed by 3x3 conv, then upsample
    branch3_downsample = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(main_path)
    branch3_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3_downsample)
    branch3_upsample = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=2, padding='same')(branch3_conv)

    # Concatenate outputs of all branches
    concatenated = Concatenate()([branch1, branch2_upsample, branch3_upsample])

    # 1x1 Convolution layer to form the main path output
    main_path_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)

    # Branch path processing
    branch_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Fusion of main path and branch path outputs
    fused_output = Add()([main_path_output, branch_path])

    # Flatten and Fully connected layer for classification
    flatten_layer = Flatten()(fused_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model