from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    model = keras.Sequential()
    
    # First Convolutional Layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    
    # Second Convolutional Layer
    x = model.output
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    x = layers.concatenate([x, model.output], axis=3)  

    # Third Convolutional Layer
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    x = layers.concatenate([x, model.output], axis=3) 

    # Flatten and Dense Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax')) 

    return model