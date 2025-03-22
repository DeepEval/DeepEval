import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Concatenate, BatchNormalization, Flatten, Dense, Conv2DTranspose
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

    # Second part: Feature Enhancement
    y = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    y = Dropout(0.5)(y)
    y = Conv2D(256, (3, 3), activation='relu', padding='same')(y)

    # Third part: Spatial Information Restoration
    z = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(y)
    z = Concatenate()([z, x])
    z = Conv2D(128, (3, 3), activation='relu', padding='same')(z)
    z = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(z)
    z = Concatenate()([z, x])
    z = Conv2D(64, (3, 3), activation='relu', padding='same')(z)

    # Output part
    output_layer = Conv2D(10, (1, 1), activation='sigmoid')(z)
    output_layer = Flatten()(output_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])