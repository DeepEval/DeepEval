import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Dropout

def dl_model():
    model = Sequential()
    
    # Add the average pooling layer with a 5x5 window and a 3x3 stride
    model.add(AveragePooling2D(pool_size=(5, 5), strides=3, input_shape=(28, 28, 1)))
    
    # Add a 1x1 convolutional layer
    model.add(Conv2D(filters=32, kernel_size=(1, 1), activation='relu'))
    
    # Flatten the feature maps
    model.add(Flatten())
    
    # Add the first fully connected layer
    model.add(Dense(256, activation='relu'))
    
    # Add a dropout layer to mitigate overfitting
    model.add(Dropout(0.5))
    
    # Add the second fully connected layer
    model.add(Dense(128, activation='relu'))
    
    # Add a dropout layer to mitigate overfitting
    model.add(Dropout(0.5))
    
    # Add the output layer with 10 classes and a softmax activation function
    model.add(Dense(10, activation='softmax'))
    
    return model

# Example usage:
# model = dl_model()
# model.summary()