import keras
from keras.layers import Input, DepthwiseConv2D, Conv2D, Dropout, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def branch(input_tensor):
        # Depthwise separable convolution
        depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        conv_1x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(depthwise_conv)
        dropout1 = Dropout(0.5)(conv_1x1)  # Dropout after the 1x1 convolution
        return dropout1

    # Create three branches
    branch1_output = branch(input_layer)
    branch2_output = branch(input_layer)
    branch3_output = branch(input_layer)

    # Concatenate the outputs of the three branches
    concatenated = Concatenate()([branch1_output, branch2_output, branch3_output])
    
    # Flatten the concatenated output
    flatten_layer = Flatten()(concatenated)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dropout2 = Dropout(0.5)(dense1)  # Dropout after the first dense layer
    dense2 = Dense(units=64, activation='relu')(dropout2)
    
    # Final output layer for classification
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model