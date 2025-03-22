import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    inputs = layers.Input(shape=(32, 32, 3))

    # Initial convolutional layer
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)

    # Basic block 1
    branch1 = layers.Conv2D(16, (1, 1), activation='relu')(inputs)  
    y = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    y = layers.BatchNormalization()(y)
    
    y = layers.add([y, branch1]) 

    # Basic block 2
    branch2 = layers.Conv2D(16, (1, 1), activation='relu')(y)
    y = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(y)
    y = layers.BatchNormalization()(y)

    y = layers.add([y, branch2])

    # Branch extraction
    branch3 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(y)

    # Feature fusion
    y = layers.add([y, branch3]) 

    # Average pooling and flattening
    y = layers.AveragePooling2D((8, 8))(y)
    y = layers.Flatten()(y)

    # Output layer
    outputs = layers.Dense(10, activation='softmax')(y) 

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model