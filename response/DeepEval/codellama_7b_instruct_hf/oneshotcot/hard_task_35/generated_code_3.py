import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1
    branch1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    branch1 = GlobalAveragePooling2D()(branch1)
    branch1 = Dense(32, activation='relu')(branch1)
    branch1 = Dense(16, activation='softmax')(branch1)

    # Branch 2
    branch2 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    branch2 = GlobalAveragePooling2D()(branch2)
    branch2 = Dense(32, activation='relu')(branch2)
    branch2 = Dense(16, activation='softmax')(branch2)

    # Concatenate branches
    concatenated = keras.layers.concatenate([branch1, branch2])

    # Flatten and fully connected layers
    flattened = Flatten()(concatenated)
    fc1 = Dense(128, activation='relu')(flattened)
    fc2 = Dense(64, activation='relu')(fc1)
    output_layer = Dense(10, activation='softmax')(fc2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model