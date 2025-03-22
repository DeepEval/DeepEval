from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_tensor = keras.Input(shape=(28, 28, 1))  # Assuming grayscale MNIST

    # First convolutional block
    x1 = layers.Conv2D(32, (3, 3), activation='relu')(input_tensor)
    x1 = layers.MaxPooling2D((2, 2))(x1)

    # Second convolutional block
    x2 = layers.Conv2D(64, (3, 3), activation='relu')(input_tensor)
    x2 = layers.MaxPooling2D((2, 2))(x2)

    # Combine outputs
    combined = layers.Add()([input_tensor, x1, x2])

    # Flatten and classify
    x = layers.Flatten()(combined)
    output = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=input_tensor, outputs=output)
    return model