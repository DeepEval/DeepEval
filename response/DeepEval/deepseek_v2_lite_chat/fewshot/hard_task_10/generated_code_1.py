import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Concatenate, Flatten, Dense

def dl_model():

    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Feature extraction path 1: 1x1 convolution
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Feature extraction path 2: sequence of convolutions
    conv2 = Conv2D(filters=64, kernel_size=(1, 7), padding='same', activation='relu')(input_layer)
    conv3 = Conv2D(filters=64, kernel_size=(7, 1), padding='same', activation='relu')(conv2)

    # Concatenate the outputs of both paths
    concat = Concatenate()([conv1, conv3])

    # 1x1 convolution to align dimensions with input image's channel
    conv4 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(concat)

    # Branch path
    branch_input = Input(shape=(32, 32, 3))
    branch = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch_input)
    add = Add()([conv4, branch])

    # Classification layer
    flatten = Flatten()(add)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=[input_layer, branch_input], outputs=dense2)

    return model