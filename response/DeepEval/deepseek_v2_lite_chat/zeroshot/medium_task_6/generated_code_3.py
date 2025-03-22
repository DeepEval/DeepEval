from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Concatenate

def dl_model():
    # Input shape
    input_shape = (24, 24, 3)  # CIFAR-10 images are 32x32, but the task does not specify a specific size

    # Input layer
    inputs = Input(shape=input_shape)

    # Initial convolution
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # First block
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Second block
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Third block
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Concatenate all features
    x = Concatenate()([x, inputs])

    # Flatten and fully connected layers
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    outputs = Dense(10, activation='softmax')(x)  # Assuming 10 classes for CIFAR-10

    # Model
    model = Model(inputs=inputs, outputs=outputs)

    return model