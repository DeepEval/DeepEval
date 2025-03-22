import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, Dropout, Concatenate
from tensorflow.keras.models import Model

def dl_model():
    input_tensor = Input(shape=(32, 32, 3))

    # Stage 1: Downsampling
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    # Stage 2: Feature Extraction
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.25)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.25)(x)

    # Stage 3: Upsampling and Reconstruction
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Concatenate()([x, input_tensor])
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Concatenate()([x, input_tensor])
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)

    # Output Layer
    output_tensor = Conv2D(10, (1, 1), activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model