import keras
from keras.layers import Input, DepthwiseConv2D, Conv2D, Dropout, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    
    # Branch 1
    branch1 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(branch1)
    branch1 = Dropout(rate=0.25)(branch1)

    # Branch 2
    branch2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Dropout(rate=0.25)(branch2)

    # Branch 3
    branch3 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    branch3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Dropout(rate=0.25)(branch3)

    # Concatenate outputs from the three branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Flatten the concatenated output
    flatten_layer = Flatten()(concatenated)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense1 = Dropout(rate=0.5)(dense1)
    dense2 = Dense(units=64, activation='relu')(dense1)
    dense2 = Dropout(rate=0.5)(dense2)

    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model