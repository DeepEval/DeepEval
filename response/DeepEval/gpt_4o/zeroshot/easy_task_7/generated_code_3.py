from tensorflow.keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    # First block
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = Dropout(0.25)(x)

    # Second block
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.25)(x)

    # Another Conv2D layer to restore the number of channels
    x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)

    # Branch path (direct connection from input)
    branch = input_layer

    # Combining both paths using addition
    combined = Add()([x, branch])

    # Flattening layer
    flattened = Flatten()(combined)

    # Fully connected layer for classification
    output_layer = Dense(10, activation='softmax')(flattened)

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model