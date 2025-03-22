import keras
from keras.layers import Input, Dense, GlobalAveragePooling2D, Flatten, Concatenate, Reshape

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1
    branch1 = GlobalAveragePooling2D()(input_layer)
    branch1 = Dense(64, activation='relu')(branch1)
    branch1 = Dense(32, activation='relu')(branch1)
    branch1 = Reshape((32, 32, 3))(branch1)

    # Branch 2
    branch2 = GlobalAveragePooling2D()(input_layer)
    branch2 = Dense(64, activation='relu')(branch2)
    branch2 = Dense(32, activation='relu')(branch2)
    branch2 = Reshape((32, 32, 3))(branch2)

    # Concatenate branches
    concatenated = Concatenate()([branch1, branch2])

    # Flatten and fully connected layers
    flattened = Flatten()(concatenated)
    dense1 = Dense(64, activation='relu')(flattened)
    dense2 = Dense(10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model