import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, AveragePooling2D, Conv2DTranspose, Concatenate, Add, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels (RGB)

    # Main path
    main_path_conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Split into branches
    # Branch 1: 3x3 convolution for local feature extraction
    branch1 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(main_path_conv1x1)

    # Branch 2: Average pooling, then 3x3 convolution
    branch2_avgpool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(main_path_conv1x1)
    branch2_conv3x3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(branch2_avgpool)
    branch2_upsample = Conv2DTranspose(filters=128, kernel_size=(2, 2), strides=2, padding='same')(branch2_conv3x3)

    # Branch 3: Average pooling, then 3x3 convolution
    branch3_avgpool = AveragePooling2D(pool_size=(4, 4), strides=4, padding='same')(main_path_conv1x1)
    branch3_conv3x3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(branch3_avgpool)
    branch3_upsample = Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=4, padding='same')(branch3_conv3x3)

    # Concatenate outputs of branches
    concatenated = Concatenate()([branch1, branch2_upsample, branch3_upsample])

    # 1x1 convolution to finalize main path output
    main_path_output = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(concatenated)

    # Branch path processing
    branch_path_output = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Fusion of main path and branch path using addition
    final_output = Add()([main_path_output, branch_path_output])

    # Flattening and classification
    flatten_layer = Flatten()(final_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model