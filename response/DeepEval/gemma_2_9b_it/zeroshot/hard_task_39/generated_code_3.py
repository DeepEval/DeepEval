from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_tensor = keras.Input(shape=(28, 28, 1))

    # Block 1: Max Pooling with varying scales
    x = layers.MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(input_tensor)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(x)
    x = layers.Flatten()(x) 
    x = layers.Concatenate()([x, x, x])

    # Fully connected layer and reshape for Block 2
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Reshape((1, 128))(x)

    # Block 2: Feature Extraction Branches
    branch1 = layers.Conv2D(32, kernel_size=(1, 1), activation='relu')(x)
    branch2 = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    branch3 = layers.Conv2D(32, kernel_size=(5, 5), activation='relu')(x)
    branch4 = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = layers.Concatenate()([branch1, branch2, branch3, branch4])

    # Final Classification Layers
    x = layers.Flatten()(x)
    x = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=input_tensor, outputs=x)

    return model