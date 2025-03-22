from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    model = keras.Sequential()
    
    # Convolutional Layer 1
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    
    # Max Pooling Layer 1
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Convolutional Layer 2
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    
    # Max Pooling Layer 2
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Flatten the output
    model.add(layers.Flatten())

    # Fully Connected Layer 1
    model.add(layers.Dense(128, activation='relu'))
    
    # Output Layer
    model.add(layers.Dense(10, activation='softmax'))

    return model