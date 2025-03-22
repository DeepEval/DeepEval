import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, DepthwiseConv2D, Flatten, Dense, Dropout

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def specialized_block(input_tensor):
        # Depthwise separable convolutional layer
        depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # 1x1 convolutional layer
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv)
        # Dropout layer to mitigate overfitting
        dropout1 = Dropout(0.25)(conv1x1)
        return dropout1

    # Apply the specialized block to each branch
    branch1 = specialized_block(input_layer)
    branch2 = specialized_block(input_layer)
    branch3 = specialized_block(input_layer)

    # Concatenate the outputs from the three branches
    concatenated_output = Concatenate()([branch1, branch2, branch3])

    # Flatten the concatenated output
    flattened_output = Flatten()(concatenated_output)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flattened_output)
    dropout2 = Dropout(0.5)(dense1)
    output_layer = Dense(units=10, activation='softmax')(dropout2)

    # Construct and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model