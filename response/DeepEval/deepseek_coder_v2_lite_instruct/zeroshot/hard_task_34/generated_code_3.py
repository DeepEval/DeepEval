import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, ReLU, Add, Flatten, Dense

def dl_model():
    # Define the input layer
    inputs = Input(shape=(28, 28, 1))

    # Main path
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = ReLU()(x)
    x = SeparableConv2D(64, (3, 3), padding='same')(x)
    x = ReLU()(x)
    x = SeparableConv2D(128, (3, 3), padding='same')(x)
    x = ReLU()(x)

    # Branch path
    y = Conv2D(128, (3, 3), padding='same')(inputs)
    y = ReLU()(y)

    # Feature fusion
    z = Add()([x, y])

    # Flatten the features
    z = Flatten()(z)

    # Fully connected layer
    outputs = Dense(10, activation='softmax')(z)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
if __name__ == "__main__":
    model = dl_model()
    model.summary()