import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Branch 1
    branch1 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    branch1 = DepthwiseConv2D((3, 3), activation='relu')(branch1)
    branch1 = Conv2D(32, (1, 1), activation='relu')(branch1)
    branch1 = Flatten()(branch1)

    # Branch 2
    branch2 = Conv2D(64, (1, 1), activation='relu')(input_layer)
    branch2 = DepthwiseConv2D((5, 5), activation='relu')(branch2)
    branch2 = Conv2D(64, (1, 1), activation='relu')(branch2)
    branch2 = Flatten()(branch2)

    # Branch 3
    branch3 = Conv2D(128, (1, 1), activation='relu')(input_layer)
    branch3 = DepthwiseConv2D((7, 7), activation='relu')(branch3)
    branch3 = Conv2D(128, (1, 1), activation='relu')(branch3)
    branch3 = Flatten()(branch3)

    # Concatenate branches
    concatenated = keras.layers.concatenate([branch1, branch2, branch3])

    # Add a fully connected layer
    flattened = Flatten()(concatenated)
    output_layer = Dense(10, activation='softmax')(flattened)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model