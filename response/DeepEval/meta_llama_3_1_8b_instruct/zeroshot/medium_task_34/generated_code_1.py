# Import necessary packages
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, UpSampling2D, concatenate, Add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow as tf

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # Convolutional and max-pooling layers (Part 1)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Processing layers (Part 2)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.2)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    # Convolutional and transposed convolutional layers with skip connections (Part 3)
    x = UpSampling2D(size=(2, 2))(x)
    x = concatenate([x, Conv2D(128, (3, 3), activation='relu', padding='same')(inputs)], axis=-1)
    x = Conv2DTranspose(64, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
    x = concatenate([x, Conv2D(64, (3, 3), activation='relu', padding='same')(x)], axis=-1)
    x = Conv2DTranspose(32, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
    x = concatenate([x, Conv2D(32, (3, 3), activation='relu', padding='same')(x)], axis=-1)

    # Output layer
    x = Conv2D(10, (1, 1), activation='softmax')(x)

    # Define the model
    model = Model(inputs=inputs, outputs=x)

    return model