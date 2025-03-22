import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Concatenate, BatchNormalization, Flatten, Dense, Conv2DTranspose
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Feature Extraction Part
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    # Feature Enhancement Part
    y = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    y = Dropout(0.5)(y)
    y = Conv2D(128, (3, 3), activation='relu', padding='same')(y)

    # Upsampling Part with Skip Connections
    z = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(y)
    z = Concatenate()([z, x])
    z = Conv2D(64, (3, 3), activation='relu', padding='same')(z)
    z = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(z)
    z = Concatenate()([z, input_layer])
    z = Conv2D(32, (3, 3), activation='relu', padding='same')(z)

    # Final Classification Part
    output_layer = Conv2D(10, (1, 1), activation='softmax')(z)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model