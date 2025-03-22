from keras.models import Model
from keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    Flatten,
    Dense,
    Dropout,
)

def dl_model():
    # Input layer
    inputs = Input(shape=(224, 224, 3))

    # Feature extraction layers
    x = inputs
    for _ in range(2):
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = AveragePooling2D()(x)

    # Additional convolutional layers and average pooling
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = AveragePooling2D()(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = AveragePooling2D()(x)

    # Flatten and fully connected layers
    x = Flatten()(x)
    for _ in range(2):
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)

    # Output layer
    outputs = Dense(1000, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model