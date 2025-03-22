import keras
from keras.layers import Input, DepthwiseConv2D, Conv2D, Dropout, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))

    def branch(input_tensor):
        # Depthwise Separable Convolution
        depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        conv_1x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(depthwise_conv)
        drop1 = Dropout(0.5)(conv_1x1)  # Dropout after 1x1 convolution
        return drop1

    # Create three branches
    branch1 = branch(input_layer)
    branch2 = branch(input_layer)
    branch3 = branch(input_layer)

    # Concatenate the outputs of the three branches
    concatenated_output = Concatenate()([branch1, branch2, branch3])

    # Flatten the concatenated output
    flatten_layer = Flatten()(concatenated_output)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    drop2 = Dropout(0.5)(dense1)  # Dropout after first dense layer
    dense2 = Dense(units=64, activation='relu')(drop2)

    # Final output layer with 10 units for classification (digits 0-9)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model