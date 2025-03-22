import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten


def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the first sequential block
    x = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)

    # Define the second sequential block
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)

    # Define the fully connected layers
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=x, outputs=x)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Return the constructed model
    return model