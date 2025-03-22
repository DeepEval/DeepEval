from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_tensor = keras.Input(shape=(32, 32, 3))  

    # 1x1 initial convolutional layer
    x = layers.Conv2D(32, kernel_size=(1, 1), activation='relu')(input_tensor)

    # Branch 1: Local Feature Extraction
    branch1 = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    branch1 = layers.Conv2D(128, kernel_size=(3, 3), activation='relu')(branch1)

    # Branch 2 & 3: Downsampling and Upsampling
    branch2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
    branch2 = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(branch2)
    branch2 = layers.Conv2DTranspose(32, kernel_size=(3, 3), strides=(2, 2), activation='relu')(branch2)

    branch3 = layers.AveragePooling2D(pool_size=(2, 2))(x)
    branch3 = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(branch3)
    branch3 = layers.Conv2DTranspose(32, kernel_size=(3, 3), strides=(2, 2), activation='relu')(branch3)

    # Concatenate branches
    x = layers.concatenate([branch1, branch2, branch3], axis=3)

    # Final 1x1 convolution
    x = layers.Conv2D(128, kernel_size=(1, 1), activation='relu')(x)

    # Flatten and output layer
    x = layers.Flatten()(x)
    output = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=input_tensor, outputs=output)
    return model