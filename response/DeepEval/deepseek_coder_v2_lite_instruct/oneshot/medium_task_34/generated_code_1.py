import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First part: Feature Extraction
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Second part: Feature Enhancement
    y = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(x)
    y = Dropout(0.5)(y)
    y = Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(y)

    # Third part: Spatial Information Restoration
    z = Conv2DTranspose(filters=128, kernel_size=(2, 2), strides=(2, 2))(y)
    z = concatenate([z, x])
    z = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2))(z)
    z = concatenate([z, x])
    z = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2))(z)
    z = concatenate([z, x])

    # Final 1x1 Convolutional Layer
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), activation='sigmoid')(z)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model