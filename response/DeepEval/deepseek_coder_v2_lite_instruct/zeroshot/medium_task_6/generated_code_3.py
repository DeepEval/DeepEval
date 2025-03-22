import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense

def dl_model():
    # Define the input shape
    inputs = Input(shape=(32, 32, 3))

    # Initial convolution
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Define the three parallel blocks
    branch1 = Conv2D(32, (3, 3), padding='same')(x)
    branch1 = BatchNormalization()(branch1)
    branch1 = ReLU()(branch1)

    branch2 = Conv2D(32, (3, 3), padding='same')(x)
    branch2 = BatchNormalization()(branch2)
    branch2 = ReLU()(branch2)

    branch3 = Conv2D(32, (3, 3), padding='same')(x)
    branch3 = BatchNormalization()(branch3)
    branch3 = ReLU()(branch3)

    # Add the outputs of the parallel blocks with the initial convolution's output
    x = Add()([x, branch1, branch2, branch3])

    # Flatten the output
    x = Flatten()(x)

    # Fully connected layers
    x = Dense(128, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
model = dl_model()
model.summary()