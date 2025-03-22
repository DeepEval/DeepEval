import keras
from keras.layers import Input, DepthwiseConv2D, Conv2D, Dropout, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def specialized_block(input_tensor):
        depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        dropout1 = Dropout(rate=0.25)(depthwise_conv)
        pointwise_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout1)
        dropout2 = Dropout(rate=0.25)(pointwise_conv)
        return dropout2

    # Create three branches using the specialized block
    branch1 = specialized_block(input_layer)
    branch2 = specialized_block(input_layer)
    branch3 = specialized_block(input_layer)

    # Concatenate the outputs of the three branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Flatten the concatenated output
    flatten = Flatten()(concatenated)

    # Two fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model