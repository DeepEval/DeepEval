import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: 1x1 Convolution
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 2: 1x1 Convolution followed by 3x3 Convolution
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)

    # Branch 3: 1x1 Convolution followed by two 3x3 Convolutions
    branch3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)

    # Concatenate the outputs of the three branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # 1x1 Convolution to adjust output dimensions
    output_branch = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)

    # Main path output
    main_path = Add()([output_branch, input_layer])  # Fusing the main path and the branch

    # Flatten and fully connected layers for classification
    flatten_layer = Flatten()(main_path)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model