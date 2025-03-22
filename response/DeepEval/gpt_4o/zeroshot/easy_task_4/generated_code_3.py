from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    inputs = Input(shape=(28, 28, 1))  # MNIST images are 28x28 with a single color channel

    # First block of layers: 3 Conv layers followed by Max Pooling
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    # Second block of layers: 4 Conv layers followed by Max Pooling
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    # Flattening the feature maps
    x = Flatten()(x)

    # Fully connected layers
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    # Output layer for classification (10 classes for digits 0-9)
    outputs = Dense(10, activation='softmax')(x)

    # Creating the model
    model = Model(inputs=inputs, outputs=outputs)

    return model