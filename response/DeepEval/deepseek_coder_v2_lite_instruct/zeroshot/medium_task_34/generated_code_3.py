import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Add, Flatten, Dense

def dl_model():
    inputs = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # First part: Feature extraction with convolutional and max-pooling layers
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # Second part: Enhancing generalization with convolutional, dropout, and convolutional layers
    y = Conv2D(256, (3, 3), activation='relu')(x)
    y = Dropout(0.5)(y)
    y = Conv2D(256, (3, 3), activation='relu')(y)

    # Third part: Upsampling with convolutional and transposed convolutional layers, using skip connections
    z = Conv2DTranspose(128, (2, 2), strides=(2, 2))(y)
    z = Add()([z, x])  # Skip connection from corresponding convolutional layer in the first part
    z = Conv2D(128, (3, 3), activation='relu')(z)

    z = Conv2DTranspose(64, (2, 2), strides=(2, 2))(z)
    z = Add()([z, x])  # Skip connection from corresponding convolutional layer in the first part
    z = Conv2D(64, (3, 3), activation='relu')(z)

    z = Conv2DTranspose(32, (2, 2), strides=(2, 2))(z)
    z = Add()([z, x])  # Skip connection from corresponding convolutional layer in the first part
    z = Conv2D(32, (3, 3), activation='relu')(z)

    # Final part: Generating the probability output
    outputs = Conv2D(10, (1, 1), activation='softmax')(z)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# Example usage:
# model = dl_model()
# model.summary()