import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, SeparableConv2D, MaxPooling2D, Concatenate, BatchNormalization, Dropout, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def branch(input_tensor):
        # Depthwise Separable Convolutional Layer
        x = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # 1x1 Convolutional Layer
        x = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        return x

    # Three branches
    branch1 = branch(input_layer)
    branch2 = branch(input_layer)
    branch3 = branch(input_layer)

    # Concatenate the outputs of the three branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Flatten the concatenated output
    flattened = Flatten()(concatenated)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model