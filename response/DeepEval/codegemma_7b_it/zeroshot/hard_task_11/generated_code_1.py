import keras
from keras.layers import Input, Conv2D, Concatenate, Dense, Add
from keras.models import Model

def dl_model():

    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # Main pathway
    main_path = Conv2D(64, (1, 1), padding='same')(inputs)
    main_path = Conv2D(64, (1, 3), padding='same')(main_path)
    main_path = Conv2D(64, (3, 1), padding='same')(main_path)

    # Parallel branch
    branch_path = Conv2D(64, (1, 1), padding='same')(inputs)
    branch_path = Conv2D(64, (1, 3), padding='same')(branch_path)
    branch_path = Conv2D(64, (3, 1), padding='same')(branch_path)

    # Concatenation
    concat_path = Concatenate()([main_path, branch_path])

    # Output layer
    output_path = Conv2D(32, (1, 1), padding='same')(concat_path)

    # Direct connection
    direct_path = Conv2D(32, (1, 1), padding='same')(inputs)

    # Fusion
    final_path = Add()([output_path, direct_path])

    # Classification layers
    flatten = keras.layers.Flatten()(final_path)
    dense1 = Dense(512, activation="relu")(flatten)
    dense2 = Dense(10, activation="softmax")(dense1)

    # Model definition
    model = Model(inputs=inputs, outputs=dense2)

    return model