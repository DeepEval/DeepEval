import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1
    branch1 = Conv2D(32, kernel_size=1, strides=1, activation='relu')(input_layer)

    # Branch 2
    branch2 = Conv2D(32, kernel_size=1, strides=1, activation='relu')(branch1)
    branch2 = Conv2D(32, kernel_size=3, strides=1, activation='relu')(branch2)

    # Branch 3
    branch3 = Conv2D(32, kernel_size=1, strides=1, activation='relu')(branch2)
    branch3 = Conv2D(32, kernel_size=3, strides=1, activation='relu')(branch3)
    branch3 = Conv2D(32, kernel_size=3, strides=1, activation='relu')(branch3)

    # Concatenate branches
    concat = Add()([branch1, branch2, branch3])

    # Adjust output dimensions
    adjusted = Conv2D(3, kernel_size=1, strides=1, activation='relu')(concat)

    # Flatten and output
    flattened = Flatten()(adjusted)
    output = Dense(10, activation='softmax')(flattened)

    model = keras.Model(inputs=input_layer, outputs=output)

    return model