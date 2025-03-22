import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    model = tf.keras.models.Sequential()

    # First Block
    model.add(layers.Lambda(lambda x: tf.split(x, 3, axis=-1), input_shape=(32, 32, 3)))
    model.add(layers.Conv2D(64, (1, 1), activation='relu'))  
    model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
    model.add(layers.Conv2D(64, (5, 5), activation='relu'))
    model.add(layers.Concatenate()) 

    # Second Block
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    
    # Branch 1
    model.add(layers.Conv2D(64, (1, 1), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
    # Branch 2
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Concatenate branches
    model.add(layers.Concatenate())
    
    # Global Average Pooling and Output
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(10, activation='softmax'))
    
    return model