import keras
from keras.layers import Input, GlobalAveragePooling2D, Flatten, Dense, Concatenate
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First branch
    branch_1 = GlobalAveragePooling2D()(input_layer)
    branch_1 = Flatten()(branch_1)
    branch_1 = Dense(64, activation='relu')(branch_1)
    branch_1 = Dense(32, activation='relu')(branch_1)

    # Second branch
    branch_2 = GlobalAveragePooling2D()(input_layer)
    branch_2 = Flatten()(branch_2)
    branch_2 = Dense(64, activation='relu')(branch_2)
    branch_2 = Dense(32, activation='relu')(branch_2)

    # Concatenate branches
    merged = Concatenate()([branch_1, branch_2])

    # Flatten and fully connected layers
    merged = Flatten()(merged)
    merged = Dense(128, activation='relu')(merged)
    merged = Dense(10, activation='softmax')(merged)

    model = Model(inputs=input_layer, outputs=merged)

    return model