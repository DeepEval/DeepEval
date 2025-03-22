import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    model = tf.keras.Sequential()

    # Specialized Feature Extraction Block
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1))) 
    model.add(layers.Conv2D(64, kernel_size=(1, 1), activation='relu'))
    model.add(layers.Conv2D(64, kernel_size=(1, 1), activation='relu'))
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Repeat the block for deeper feature extraction
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu')) 
    model.add(layers.Conv2D(128, kernel_size=(1, 1), activation='relu'))
    model.add(layers.Conv2D(128, kernel_size=(1, 1), activation='relu'))
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Global Average Pooling
    model.add(layers.GlobalAveragePooling2D())

    # Flatten and Output Layer
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax')) 

    return model