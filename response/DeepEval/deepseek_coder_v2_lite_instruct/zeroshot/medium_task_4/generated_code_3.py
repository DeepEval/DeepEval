import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Add

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Path 1
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x1)
    x1 = AveragePooling2D((2, 2))(x1)

    x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    x1 = AveragePooling2D((2, 2))(x1)

    # Path 2
    x2 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    x2 = AveragePooling2D((2, 2))(x2)

    # Addition of the two pathways
    added = Add()([x1, x2])

    # Flatten the output
    flatten = Flatten()(added)

    # Fully connected layer
    output_layer = Dense(10, activation='softmax')(flatten)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage
model = dl_model()
model.summary()