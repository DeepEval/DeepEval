import keras
from keras.layers import Input, Conv2D, Add, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    # Path 1: 1x1 Convolution
    path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Path 2: Sequence of Convolutions (1x1 -> 1x7 -> 7x1)
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    conv1x7 = Conv2D(filters=32, kernel_size=(1, 7), padding='same', activation='relu')(conv1x1)
    conv7x1 = Conv2D(filters=32, kernel_size=(7, 1), padding='same', activation='relu')(conv1x7)

    # Concatenate the outputs of both paths
    concatenated = Concatenate()([path1, conv7x1])

    # 1x1 Convolution to align output dimensions with input channels
    main_output = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='relu')(concatenated)

    # Branch directly connected to the input
    branch_output = input_layer

    # Merging the main path and the branch through addition
    merged_output = Add()([main_output, branch_output])

    # Flatten and fully connected layers for classification
    flatten_layer = Flatten()(merged_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Constructing the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model