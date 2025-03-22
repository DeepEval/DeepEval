import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Conv2DTranspose, Concatenate, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Main path
    main_path_conv1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 1: Local feature extraction
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(main_path_conv1)

    # Branch 2: Downsampling and processing
    branch2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(main_path_conv1)
    branch2_conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
    branch2_upsample = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=2, padding='same')(branch2_conv)

    # Branch 3: Downsampling and processing
    branch3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='same')(main_path_conv1)
    branch3_conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch3)
    branch3_upsample = Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=4, padding='same')(branch3_conv)

    # Concatenating outputs of all branches
    concatenated = Concatenate()([branch1, branch2_upsample, branch3_upsample])

    # Final main path layer
    main_path_output = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(concatenated)

    # Branch path to match number of channels
    branch_path_output = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Fusing main path and branch path outputs
    fused_output = Add()([main_path_output, branch_path_output])

    # Fully connected layer for classification
    flatten_layer = Flatten()(fused_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # 10 classes for CIFAR-10

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model