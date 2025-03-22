import keras
from keras.layers import Input, Conv2D, Concatenate, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Branch 1: 1x1 Convolution
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 2: 1x1 Convolution followed by 3x3 Convolution
    branch2_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2_1)

    # Branch 3: 1x1 Convolution followed by two 3x3 Convolutions
    branch3_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3_1)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3_2)

    # Concatenate the outputs of the three branches
    concatenated_branches = Concatenate()([branch1, branch2, branch3])

    # 1x1 Convolution to adjust output dimensions
    adjust_conv = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated_branches)

    # Add the adjusted result to the input
    fused_output = Add()([adjust_conv, input_layer])

    # Flatten the output
    flatten_layer = Flatten()(fused_output)

    # Fully connected layers for classification
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)  # CIFAR-10 has 10 classes

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model