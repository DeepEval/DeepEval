import keras
from keras.layers import Input, Conv2D, Add, Concatenate, MaxPool2D, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Feature extraction path 1: 1x1 convolution
    conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(input_layer)
    # Feature extraction path 2: sequence of convolutions
    conv2_1 = Conv2D(filters=64, kernel_size=(1, 7), padding='same')(input_layer)
    conv2_2 = Conv2D(filters=64, kernel_size=(7, 1), padding='same')(input_layer)
    # Concatenate the outputs of the two paths
    concat_layer = Concatenate()([conv1_1, conv2_1, conv2_2])

    # Add a branch that directly connects to the input
    branch = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(input_layer)
    # Merge the outputs of the main path and the branch through addition
    addition = Add()([concat_layer, branch])

    # 1x1 convolution to align the output dimensions with the input image's channel
    conv3_1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(concat_layer)
    conv3_1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(concat_layer)

    # Flatten the output and pass through two fully connected layers for classification
    flatten_layer = Flatten()(addition)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model