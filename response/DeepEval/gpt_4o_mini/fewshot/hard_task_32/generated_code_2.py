import keras
from keras.layers import Input, DepthwiseConv2D, Conv2D, Dropout, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def branch_block(input_tensor):
        depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        conv = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(depthwise_conv)
        drop = Dropout(0.5)(conv)
        return drop

    # Create three branches
    branch1_output = branch_block(input_layer)
    branch2_output = branch_block(input_layer)
    branch3_output = branch_block(input_layer)

    # Concatenate the outputs from the branches
    concatenated = Concatenate()([branch1_output, branch2_output, branch3_output])

    # Fully connected layers
    flatten_layer = Flatten()(concatenated)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    drop_dense = Dropout(0.5)(dense1)  # Applying dropout to the first dense layer
    output_layer = Dense(units=10, activation='softmax')(drop_dense)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model