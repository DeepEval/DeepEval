import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, SeparableConv2D

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # Main Path
    x = input_layer
    for _ in range(3):
        x = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = Concatenate(axis=-1)([x, input_layer])  # Concatenate with the input

    # Branch Path
    branch_x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)

    # Feature Fusion
    x = keras.layers.add([x, branch_x])

    # Flatten and Output
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model