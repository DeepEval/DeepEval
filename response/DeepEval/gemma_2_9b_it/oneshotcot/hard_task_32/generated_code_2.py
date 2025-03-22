import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Dropout, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # Branch 1
    branch1 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(branch1)
    branch1 = Dropout(rate=0.2)(branch1)

    # Branch 2
    branch2 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same')(input_layer)
    branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(branch2)
    branch2 = Dropout(rate=0.3)(branch2)

    # Branch 3
    branch3 = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same')(input_layer)
    branch3 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same')(branch3)
    branch3 = Dropout(rate=0.4)(branch3)

    # Concatenate branches
    merged_features = Concatenate()([branch1, branch2, branch3])

    # Flatten and fully connected layers
    flattened = Flatten()(merged_features)
    dense1 = Dense(units=128, activation='relu')(flattened)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model