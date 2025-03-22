import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Define the model
    model = models.Sequential()

    # Add a Conv2D layer with 32 filters, a kernel size of 3x3, and ReLU activation
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    
    # Add a MaxPooling2D layer with a pool size of 2x2
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Add another Conv2D layer with 64 filters, a kernel size of 3x3, and ReLU activation
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
    # Add a MaxPooling2D layer with a pool size of 2x2
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Add another Conv2D layer with 128 filters, a kernel size of 3x3, and ReLU activation
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    
    # Add a MaxPooling2D layer with a pool size of 2x2
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Add a GlobalAveragePooling2D layer to get the average of each feature map
    model.add(layers.GlobalAveragePooling2D())
    
    # Add a Flatten layer to convert the 3D output to 1D
    model.add(layers.Flatten())
    
    # Add a Dense layer with 64 units and ReLU activation
    model.add(layers.Dense(64, activation='relu'))
    
    # Add the output Dense layer with 10 units (for CIFAR-10 classes) and softmax activation
    model.add(layers.Dense(10, activation='softmax'))
    
    return model

# Example usage:
# model = dl_model()
# model.summary()