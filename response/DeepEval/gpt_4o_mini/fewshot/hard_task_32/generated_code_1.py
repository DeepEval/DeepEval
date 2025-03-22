import keras
from keras.layers import Input, DepthwiseConv2D, Conv2D, Dropout, Flatten, Concatenate, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def branch_block(input_tensor):
        # Depthwise separable convolution
        depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        # 1x1 convolution
        conv_1x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(depthwise_conv)
        # Dropout for regularization
        dropout = Dropout(rate=0.3)(conv_1x1)
        return dropout

    # Creating three branches
    branch1_output = branch_block(input_layer)
    branch2_output = branch_block(input_layer)
    branch3_output = branch_block(input_layer)

    # Concatenating the outputs of the three branches
    concatenated = Concatenate()([branch1_output, branch2_output, branch3_output])

    # Flattening the concatenated output
    flatten = Flatten()(concatenated)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dropout1 = Dropout(rate=0.5)(dense1)  # Dropout after the first dense layer
    output_layer = Dense(units=10, activation='softmax')(dropout1)  # Output layer for classification

    # Constructing the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model