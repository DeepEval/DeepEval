import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Add, Dropout, Activation

def dl_model():
    inputs = Input(shape=(32, 32, 3))

    # Encoder
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # Bottleneck
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = Activation('relu')(x)

    # Decoder
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = Add()([x, inputs])  # Skip connection
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(x)
    x = Add()([x, inputs])  # Skip connection
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = Activation('relu')(x)

    # Output
    x = Conv2D(10, (1, 1), activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)

    return model

# Example usage
model = dl_model()
model.summary()