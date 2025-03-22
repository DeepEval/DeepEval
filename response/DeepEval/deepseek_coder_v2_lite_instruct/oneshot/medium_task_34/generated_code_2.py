import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First part: Feature Extraction
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    # Second part: Generalization Enhancement
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)

    # Store the output of the second part for later use
    skip2_input = x

    # Third part: Spatial Information Restoration
    x = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(x)
    x = Concatenate()([x, skip2_input])
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = Concatenate()([x, skip2_input])
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = Concatenate()([x, skip2_input])
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    # Final part: Classification
    x = Conv2D(32, (1, 1), activation='relu')(x)
    output_layer = Dense(10, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage:
# model = dl_model()
# model.summary()