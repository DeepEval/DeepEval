from keras.layers import Input, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, AveragePooling2D
from keras.models import Model

def dl_model():
    # Define input shape
    input_shape = (28, 28, 1)

    # Define first block
    x = Input(shape=input_shape)
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x1 = MaxPooling2D((2, 2), padding='same')(x1)
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x1)
    x1 = MaxPooling2D((2, 2), padding='same')(x1)
    x1 = Flatten()(x1)
    x1 = Dropout(0.2)(x1)

    # Define second block
    x2 = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x2 = MaxPooling2D((2, 2), padding='same')(x2)
    x2 = Conv2D(64, (3, 3), activation='relu', padding='same')(x2)
    x2 = MaxPooling2D((2, 2), padding='same')(x2)
    x2 = Flatten()(x2)
    x2 = Dropout(0.2)(x2)

    # Concatenate outputs of both blocks
    x = concatenate([x1, x2], axis=1)

    # Define fully connected layers and output layer
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Create and return model
    model = Model(inputs=x, outputs=x)
    return model