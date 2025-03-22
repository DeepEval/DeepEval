import keras
from keras.layers import Conv2D, MaxPooling2D, Input, concatenate, add, Flatten, Dense

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 64))

    # Main path
    x = Conv2D(32, (1, 1), strides=(2, 2), padding='same')(inputs)
    x = Conv2D(64, (1, 1), strides=(1, 1), padding='same')(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)

    # Branch path
    branch_path = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(inputs)

    # Combine outputs
    combined = add([x, branch_path])

    # Classification layers
    x = Flatten()(combined)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)

    # Create model
    model = keras.Model(inputs=inputs, outputs=predictions)

    return model