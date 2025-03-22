from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_img = keras.Input(shape=(28, 28, 1))  # Assuming MNIST is grayscale

    # Block 1
    x1 = layers.Conv2D(32, (3, 3), activation='relu')(input_img)
    x1 = layers.MaxPooling2D((2, 2))(x1)

    # Block 2
    x2 = layers.Conv2D(64, (3, 3), activation='relu')(input_img)
    x2 = layers.MaxPooling2D((2, 2))(x2)

    # Combine outputs
    combined = layers.add([input_img, x1, x2])  

    # Flatten and classify
    x = layers.Flatten()(combined)
    output = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=input_img, outputs=output)
    return model