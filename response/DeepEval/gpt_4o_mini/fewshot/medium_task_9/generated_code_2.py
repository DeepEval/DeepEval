import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def basic_block(input_tensor):
    # Main path
    conv_main = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(input_tensor)
    batch_norm_main = BatchNormalization()(conv_main)
    relu_main = ReLU()(batch_norm_main)

    # Branch path (shortcut connection)
    branch = input_tensor

    # Feature fusion via addition
    output_tensor = Add()([relu_main, branch])
    
    return output_tensor

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Initial convolution to reduce dimensionality
    initial_conv = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(input_layer)

    # First basic block
    block1_output = basic_block(initial_conv)

    # Second basic block
    block2_output = basic_block(block1_output)

    # Branch extraction features
    branch_output = Conv2D(filters=16, kernel_size=(1, 1), padding='same')(initial_conv)

    # Combine outputs from both paths
    combined_output = Add()([block2_output, branch_output])

    # Downsample using average pooling
    pooled_output = AveragePooling2D(pool_size=(2, 2))(combined_output)

    # Flatten and fully connected layer for classification
    flatten = Flatten()(pooled_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)  # CIFAR-10 has 10 classes

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model