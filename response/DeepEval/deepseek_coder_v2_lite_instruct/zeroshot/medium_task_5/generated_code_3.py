import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    # Define the input shape
    inputs = Input(shape=(32, 32, 3))

    # Main Path
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    # Branch Path
    y = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    y = MaxPooling2D((2, 2))(y)

    # Addition Operation
    combined = Add()([x, y])

    # Flatten the combined output
    flattened = Flatten()(combined)

    # Fully Connected Layers
    fc1 = Dense(128, activation='relu')(flattened)
    outputs = Dense(10, activation='softmax')(fc1)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model