import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, AveragePooling2D, Conv2DTranspose, Concatenate, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Main Path
    main_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Split into three branches
    # Branch 1: Local feature extraction
    branch1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_conv1)

    # Branch 2: Downsample with pooling, then process and upsample
    branch2_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(main_conv1)
    branch2_conv = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2_pool)
    branch2_upsample = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same')(branch2_conv)

    # Branch 3: Another downsample with pooling, then process and upsample
    branch3_pool = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(main_conv1)
    branch3_conv = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3_pool)
    branch3_upsample = Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(4, 4), padding='same')(branch3_conv)

    # Concatenate outputs from all branches
    concatenated = Concatenate()([branch1, branch2_upsample, branch3_upsample])

    # Final convolution in the main path
    main_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)

    # Branch Path
    branch_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Fuse main path and branch path outputs
    fused_output = Add()([main_output, branch_output])

    # Flatten and classify
    flatten_layer = Flatten()(fused_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model