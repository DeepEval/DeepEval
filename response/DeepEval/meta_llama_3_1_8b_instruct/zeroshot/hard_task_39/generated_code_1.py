from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import initializers

def dl_model():
    # Block 1: Max Pooling and Concatenation
    input_data = keras.Input(shape=(28, 28, 1))
    x = layers.MaxPooling2D(pool_size=(1, 1), strides=1, padding='same')(input_data)
    x = layers.Flatten()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_data)
    x = layers.Flatten()(x)
    x = layers.Concatenate()([x, layers.MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(input_data)])
    x = layers.Flatten()(x)

    # Fully Connected Layer and Reshape
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Reshape((4, 4, 128))(x)

    # Block 2: Convolutional and Max Pooling Branches
    branch1 = layers.Conv2D(32, kernel_size=(1, 1), activation='relu', padding='same')(x)
    branch2 = layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(x)
    branch3 = layers.Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same')(x)
    branch4 = layers.MaxPooling2D(pool_size=(3, 3), strides=3, padding='same')(x)

    # Concatenate and Flatten
    x = layers.Concatenate()([branch1, branch2, branch3, branch4])
    x = layers.Flatten()(x)

    # Fully Connected Layer for Classification
    x = layers.Dense(10, activation='softmax')(x)

    # Create Model
    model = keras.Model(inputs=input_data, outputs=x)

    return model

# Create the model
model = dl_model()
model.summary()