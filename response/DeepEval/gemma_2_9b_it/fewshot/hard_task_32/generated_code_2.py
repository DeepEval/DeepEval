import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Dropout, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def branch_block(input_tensor):
        depthwise = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise)
        dropout1 = Dropout(0.2)(conv)
        dropout2 = Dropout(0.2)(dropout1)
        return dropout2

    branch1 = branch_block(input_layer)
    branch2 = branch_block(input_layer)
    branch3 = branch_block(input_layer)

    concatenated = Concatenate()([branch1, branch2, branch3])
    flatten = Flatten()(concatenated)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model