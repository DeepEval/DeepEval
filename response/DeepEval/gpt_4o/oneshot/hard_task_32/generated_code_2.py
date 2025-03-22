import keras
from keras.layers import Input, SeparableConv2D, Conv2D, Dropout, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        # Depthwise separable convolution
        sep_conv = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        dropout1 = Dropout(rate=0.3)(sep_conv)
        
        # 1x1 Convolution
        conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(dropout1)
        dropout2 = Dropout(rate=0.3)(conv1x1)
        
        return dropout2

    # Create three branches
    branch1 = block(input_layer)
    branch2 = block(input_layer)
    branch3 = block(input_layer)

    # Concatenate outputs from the branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Flatten the concatenated output
    flatten_layer = Flatten()(concatenated)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model