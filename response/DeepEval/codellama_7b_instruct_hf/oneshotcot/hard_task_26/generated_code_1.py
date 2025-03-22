import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    x = Conv2D(32, (3, 3), padding='same')(input_layer)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = Concatenate()([x, x, x])
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Branch path
    branch_input = Input(shape=(32, 32, 3))
    branch_x = Conv2D(32, (3, 3), padding='same')(branch_input)
    branch_x = MaxPooling2D((2, 2))(branch_x)
    branch_x = Conv2D(128, (3, 3), padding='same')(branch_x)
    branch_x = Concatenate()([branch_x, branch_x, branch_x])
    branch_x = BatchNormalization()(branch_x)
    branch_x = Flatten()(branch_x)
    branch_x = Dense(128, activation='relu')(branch_x)
    branch_x = Dense(10, activation='softmax')(branch_x)

    # Add and process the outputs from both paths
    x = keras.layers.add([x, branch_x])
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Define the model
    model = keras.models.Model(inputs=[input_layer, branch_input], outputs=x)

    return model