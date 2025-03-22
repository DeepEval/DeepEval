import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Flatten, Dense, GlobalAveragePooling2D

def dl_model():
    # Input layer
    inputs = Input(shape=(28, 28, 1))

    # First specialized block
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = AveragePooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # Second specialized block
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = AveragePooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)

    # Flatten layer
    x = Flatten()(x)

    # Fully connected layer
    outputs = Dense(10, activation='softmax')(x)

    # Model
    model = Model(inputs=inputs, outputs=outputs)

    return model