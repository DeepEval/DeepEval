import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main pathway
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Parallel branch
    y = Conv2D(16, (1, 1), activation='relu')(input_layer)
    y = MaxPooling2D((2, 2))(y)
    y = Conv2D(32, (1, 3), activation='relu')(y)
    y = MaxPooling2D((2, 2))(y)
    y = Conv2D(64, (3, 1), activation='relu')(y)
    y = MaxPooling2D((2, 2))(y)
    y = Flatten()(y)
    y = Dense(128, activation='relu')(y)
    y = Dense(10, activation='softmax')(y)

    # Concat and add
    x = keras.layers.Concatenate()([x, y])
    x = Dense(10, activation='softmax')(x)

    # Model
    model = Model(inputs=input_layer, outputs=x)

    return model