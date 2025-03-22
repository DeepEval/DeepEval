import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, MaxPooling2D, Concatenate, BatchNormalization, Dropout, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Define a specialized block for each branch
    def specialized_block(input_tensor):
        # Depthwise separable convolutional layer
        depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # 1x1 convolutional layer
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv)
        # Dropout to mitigate overfitting
        dropout1 = Dropout(0.25)(conv1x1)
        dropout2 = Dropout(0.25)(depthwise_conv)
        return dropout1, dropout2

    # Apply the specialized block to each branch
    branch1_output, branch2_output = specialized_block(input_layer)
    branch3_output, branch4_output = specialized_block(input_layer)

    # Concatenate the outputs from the three branches
    concatenated = Concatenate()([branch1_output, branch2_output, branch3_output, branch4_output])

    # Batch normalization and flatten
    batch_norm = BatchNormalization()(concatenated)
    flatten_layer = Flatten()(batch_norm)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model