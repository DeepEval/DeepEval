import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense

def dl_model():
    # Define the input shape
    inputs = Input(shape=(32, 32, 3))

    # First block
    x1 = Conv2D(32, (3, 3), padding='same')(inputs)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)

    # Second block
    x2 = Conv2D(64, (3, 3), padding='same')(x1)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)

    # Third block
    x3 = Conv2D(128, (3, 3), padding='same')(x2)
    x3 = BatchNormalization()(x3)
    x3 = ReLU()(x3)

    # Parallel branch processing the input directly
    y = Conv2D(32, (3, 3), padding='same')(inputs)
    y = BatchNormalization()(y)
    y = ReLU()(y)

    # Adding outputs from all paths
    added = Add()([x1, x2, x3, y])

    # Flatten and pass through two fully connected layers
    flattened = Flatten()(added)
    outputs = Dense(10, activation='softmax')(flattened)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
model = dl_model()
model.summary()