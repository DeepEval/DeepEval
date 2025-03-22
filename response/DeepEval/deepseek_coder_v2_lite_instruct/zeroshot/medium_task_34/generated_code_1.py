import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Add, Flatten, Dense

def dl_model():
    input_shape = (32, 32, 3)
    inputs = Input(shape=input_shape)

    # First part: Sequential convolutional layers and max-pooling layers
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # Second part: Additional convolutional layers, Dropout, and convolutional layers
    y = Conv2D(256, (3, 3), activation='relu')(x)
    y = Dropout(0.5)(y)
    y = Conv2D(256, (3, 3), activation='relu')(y)

    # Skip connections
    z = Add()([x, y])

    # Third part: Upsampling with convolutional layers and transposed convolutional layers
    z = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(z)
    z = Conv2D(128, (3, 3), activation='relu')(z)
    z = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(z)
    z = Conv2D(64, (3, 3), activation='relu')(z)
    z = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(z)
    z = Conv2D(32, (3, 3), activation='relu')(z)

    # Final part: 1x1 convolutional layer for classification
    outputs = Conv2D(10, (1, 1), activation='softmax')(z)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# Example usage
model = dl_model()
model.summary()