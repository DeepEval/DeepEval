import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))

    # Main Path
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = SeparableConv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = SeparableConv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    # Branch Path
    branch = Conv2D(128, (1, 1), activation='relu', padding='same')(input_layer)

    # Concatenate Paths
    x = Concatenate()([x, branch])

    x = Flatten()(x)
    output_layer = Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model