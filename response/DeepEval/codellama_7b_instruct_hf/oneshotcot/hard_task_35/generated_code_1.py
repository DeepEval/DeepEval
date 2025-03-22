import keras
from keras.layers import Input, GlobalAveragePooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1
    branch_1 = input_layer
    branch_1 = GlobalAveragePooling2D()(branch_1)
    branch_1 = Dense(units=64, activation='relu')(branch_1)
    branch_1 = Dense(units=128, activation='relu')(branch_1)
    branch_1 = Flatten()(branch_1)

    # Branch 2
    branch_2 = input_layer
    branch_2 = GlobalAveragePooling2D()(branch_2)
    branch_2 = Dense(units=64, activation='relu')(branch_2)
    branch_2 = Dense(units=128, activation='relu')(branch_2)
    branch_2 = Flatten()(branch_2)

    # Concatenate branches
    concatenated_branches = keras.layers.concatenate([branch_1, branch_2], axis=1)

    # Flatten and dense layers
    flattened = Flatten()(concatenated_branches)
    dense_1 = Dense(units=128, activation='relu')(flattened)
    dense_2 = Dense(units=64, activation='relu')(dense_1)
    output_layer = Dense(units=10, activation='softmax')(dense_2)

    # Create and return model
    model = Model(inputs=input_layer, outputs=output_layer)
    return model