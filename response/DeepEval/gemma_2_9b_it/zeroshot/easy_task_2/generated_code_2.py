import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout

def dl_model():
    model = tf.keras.Sequential()

    # Feature Extraction Layers
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(AveragePooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(AveragePooling2D((2, 2)))

    # Additional Convolutional Layers
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(Conv2D(512, (3, 3), activation='relu'))

    # Further Dimensionality Reduction
    model.add(AveragePooling2D((2, 2)))
    model.add(Flatten())

    # Fully Connected Layers with Dropout
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))  

    return model