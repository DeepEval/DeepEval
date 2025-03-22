from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    model = keras.Sequential()
    
    # Input layer
    model.add(layers.Input(shape=(28, 28, 1))) 

    # 1x1 Convolutional layer to increase dimensionality
    model.add(layers.Conv2D(32, (1, 1), activation='relu')) 

    # 3x3 Depthwise Separable Convolutional layer for feature extraction
    model.add(layers.DepthwiseConv2D(kernel_size=(3, 3), activation='relu')) 

    # 1x1 Convolutional layer to reduce dimensionality
    model.add(layers.Conv2D(32, (1, 1), activation='relu')) 

    # Add output of convolutional layers to original input
    model.add(layers.Add()) 

    # Flatten the output
    model.add(layers.Flatten())

    # Fully connected layer
    model.add(layers.Dense(10, activation='softmax'))

    return model