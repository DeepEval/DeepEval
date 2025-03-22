import keras
from keras.layers import Input, Conv2D, SeparableConv2D, MaxPooling2D, Concatenate, Dropout, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # Branch 1
    branch1 = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(branch1)
    branch1 = Dropout(0.25)(branch1)

    # Branch 2
    branch2 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(branch2)
    branch2 = Dropout(0.25)(branch2)

    # Branch 3
    branch3 = SeparableConv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same')(branch3)
    branch3 = Dropout(0.25)(branch3)

    # Concatenate branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Flatten and dense layers
    flattened = Flatten()(concatenated)
    dense1 = Dense(units=128, activation='relu')(flattened)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model