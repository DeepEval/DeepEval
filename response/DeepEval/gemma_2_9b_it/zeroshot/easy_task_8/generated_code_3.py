import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    model = tf.keras.Sequential()
    
    # Depthwise Separable Convolutional Layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                            use_depthwise=True, input_shape=(28, 28, 1)))
    model.add(layers.Conv2D(32, (1, 1), activation='relu', padding='same'))
    model.add(layers.Dropout(0.25)) 
    
    # Second Convolutional Block
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                            use_depthwise=True))
    model.add(layers.Conv2D(64, (1, 1), activation='relu', padding='same'))
    model.add(layers.Dropout(0.25)) 
    
    # Flatten and Fully Connected Layer
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))
    
    return model