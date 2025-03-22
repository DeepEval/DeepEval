import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1
    branch1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    branch1 = MaxPooling2D((2, 2))(branch1)
    branch1 = GlobalAveragePooling2D()(branch1)
    branch1 = Flatten()(branch1)
    branch1 = Dense(64, activation='relu')(branch1)
    branch1 = Dense(32, activation='relu')(branch1)

    # Branch 2
    branch2 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    branch2 = MaxPooling2D((2, 2))(branch2)
    branch2 = GlobalAveragePooling2D()(branch2)
    branch2 = Flatten()(branch2)
    branch2 = Dense(64, activation='relu')(branch2)
    branch2 = Dense(32, activation='relu')(branch2)

    # Concatenate the outputs from both branches
    merged = keras.layers.concatenate([branch1, branch2])

    # Flatten layer
    flattened = Flatten()(merged)

    # Fully connected layer
    output = Dense(10, activation='softmax')(flattened)

    # Create and return the model
    model = keras.Model(inputs=input_layer, outputs=output)
    return model