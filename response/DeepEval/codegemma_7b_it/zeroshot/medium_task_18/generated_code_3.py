import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, concatenate

def dl_model():
    # Input layer
    inputs = keras.Input(shape=(32, 32, 3))

    # Feature extraction at different scales
    tower_1 = Conv2D(32, (1, 1), padding='same')(inputs)
    tower_1 = MaxPooling2D((2, 2), padding='same')(tower_1)
    tower_1 = Conv2D(64, (1, 1), padding='same')(tower_1)
    tower_1 = MaxPooling2D((2, 2), padding='same')(tower_1)
    tower_1 = Conv2D(128, (1, 1), padding='same')(tower_1)
    tower_1 = MaxPooling2D((2, 2), padding='same')(tower_1)

    tower_2 = Conv2D(32, (3, 3), padding='same')(inputs)
    tower_2 = MaxPooling2D((2, 2), padding='same')(tower_2)
    tower_2 = Conv2D(64, (3, 3), padding='same')(tower_2)
    tower_2 = MaxPooling2D((2, 2), padding='same')(tower_2)
    tower_2 = Conv2D(128, (3, 3), padding='same')(tower_2)
    tower_2 = MaxPooling2D((2, 2), padding='same')(tower_2)

    tower_3 = Conv2D(32, (5, 5), padding='same')(inputs)
    tower_3 = MaxPooling2D((2, 2), padding='same')(tower_3)
    tower_3 = Conv2D(64, (5, 5), padding='same')(tower_3)
    tower_3 = MaxPooling2D((2, 2), padding='same')(tower_3)
    tower_3 = Conv2D(128, (5, 5), padding='same')(tower_3)
    tower_3 = MaxPooling2D((2, 2), padding='same')(tower_3)

    # Concatenate features
    merged = concatenate([tower_1, tower_2, tower_3])

    # Classification
    flatten = Flatten()(merged)
    fc1 = Dense(512, activation='relu')(flatten)
    fc2 = Dense(10, activation='softmax')(fc1)

    # Model definition
    model = keras.Model(inputs, fc2)

    return model

model = dl_model()