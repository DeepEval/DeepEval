import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense, BatchNormalization, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: 1x1 Convolution
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 2: 1x1 Convolution followed by 3x3 Convolution
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)

    # Branch 3: 1x1 Convolution followed by two 3x3 Convolutions
    branch3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch3)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch3)

    # Concatenate the outputs of the three branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # 1x1 Convolution to adjust output dimensions
    adjustment = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='relu')(concatenated)

    # Adding the adjusted output back to the original input
    merged = Add()([input_layer, adjustment])

    # Flatten the output and create fully connected layers
    flatten_layer = Flatten()(merged)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model