from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    
    model = keras.Sequential()
    
    # First Convolutional Layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    
    # Second Convolutional Layer
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
    # Third Convolutional Layer
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))

    # Concatenate outputs of each convolutional layer along the channel dimension
    output = layers.concatenate([model.layers[-1].output, model.layers[-2].output, model.layers[-3].output], axis=-1)

    # Flatten the concatenated output
    model.add(layers.Flatten())

    # First Fully Connected Layer
    model.add(layers.Dense(128, activation='relu'))

    # Output Layer
    model.add(layers.Dense(10, activation='softmax')) 

    return model