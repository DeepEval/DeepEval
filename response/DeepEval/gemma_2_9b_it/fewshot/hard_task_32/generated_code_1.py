import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Dropout, Concatenate, Dense, Flatten

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def branch_block(input_tensor):
        x = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
        x = Dropout(0.25)(x)
        return x

    branch1 = branch_block(input_layer)
    branch2 = branch_block(input_layer)
    branch3 = branch_block(input_layer)

    merged = Concatenate()([branch1, branch2, branch3])

    flatten = Flatten()(merged)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model