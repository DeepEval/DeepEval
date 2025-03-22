from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_tensor = keras.Input(shape=(28, 28, 1))  

    # First convolutional block
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)
    x = layers.MaxPooling2D((2, 2))(x)

    # Second convolutional block
    x_branch = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_tensor)
    x_branch = layers.MaxPooling2D((2, 2))(x_branch)

    # Combine outputs from both branches and input
    x = layers.Add()([x, x_branch])

    # Flatten and classify
    x = layers.Flatten()(x)
    output_tensor = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=input_tensor, outputs=output_tensor)
    return model