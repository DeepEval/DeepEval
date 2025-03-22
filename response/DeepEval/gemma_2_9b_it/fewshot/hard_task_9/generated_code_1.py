import keras
from keras.layers import Input, Conv2D, Add, Concatenate, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: 1x1 convolution
    branch1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 2: 1x1 convolution followed by 3x3 convolution
    branch2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)

    # Branch 3: 1x1 convolution followed by two 3x3 convolutions
    branch3 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)

    # Concatenate outputs of three branches
    concatenated_branches = Concatenate()([branch1, branch2, branch3])

    # 1x1 convolution to adjust dimensions
    concatenated_branches = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated_branches)

    # Add branch output to main path
    main_path = Add()([input_layer, concatenated_branches])

    # Flatten and fully connected layers
    flatten_layer = Flatten()(main_path)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model