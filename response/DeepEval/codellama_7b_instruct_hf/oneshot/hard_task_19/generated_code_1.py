import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = GlobalAveragePooling2D()(x)

    # Branch path
    branch_input = Input(shape=(32, 32, 3))
    branch_output = Dense(64, activation='relu')(branch_input)
    branch_output = Dense(32, activation='relu')(branch_output)

    # Concatenate main and branch paths
    output = keras.layers.concatenate([x, branch_output])

    # Additional layers
    output = Flatten()(output)
    output = Dense(128, activation='relu')(output)
    output = Dense(10, activation='softmax')(output)

    # Create and return model
    model = keras.Model(inputs=[input_layer, branch_input], outputs=output)
    return model