from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model

def block(input_tensor):
    x = BatchNormalization()(input_tensor)
    x = ReLU()(x)
    x = Conv2D(32, (3, 3), padding='same')(x)  # 32 filters as an example
    x = Concatenate()([input_tensor, x])
    return x

def pathway(input_tensor):
    x = input_tensor
    for _ in range(3):  # Repeat the block structure three times
        x = block(x)
    return x

def dl_model():
    input_shape = (28, 28, 1)  # MNIST images are 28x28 with 1 channel

    inputs = Input(shape=input_shape)

    # First pathway
    x1 = pathway(inputs)

    # Second pathway
    x2 = pathway(inputs)

    # Concatenate the outputs of the two pathways
    merged = Concatenate()([x1, x2])

    # Classification layers
    x = Flatten()(merged)
    x = Dense(128, activation='relu')(x)  # Example with 128 neurons
    outputs = Dense(10, activation='softmax')(x)  # 10 classes for MNIST

    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
model = dl_model()
model.summary()