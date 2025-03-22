import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First branch
    x = Conv2D(32, (1, 1), activation='relu')(input_layer)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Second branch
    x = Conv2D(64, (1, 1), activation='relu')(input_layer)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Third branch
    x = Conv2D(128, (1, 1), activation='relu')(input_layer)
    x = Conv2D(128, (5, 5), activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Fourth branch
    x = MaxPooling2D((3, 3), strides=(2, 2))(input_layer)
    x = Conv2D(256, (1, 1), activation='relu')(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Concat branches
    x = Concatenate()([x, x, x])

    # Flatten features
    x = Flatten()(x)

    # Dense layers
    x = Dense(512, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Create and compile model
    model = keras.Model(inputs=input_layer, outputs=x)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model