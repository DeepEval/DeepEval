from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))  

    # Initial Convolutional Layer
    x = layers.Conv2D(16, kernel_size=(3, 3), activation='relu')(inputs)

    # Basic Block
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(16, kernel_size=(3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Branch
    shortcut = x
    x = layers.Conv2D(16, kernel_size=(1, 1))(x) 
    x = layers.BatchNormalization()(x)
    x = layers.add([shortcut, x]) 

    # Level 1 Residual Connection
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(16, kernel_size=(3, 3), activation='relu')(x) 
    x = layers.BatchNormalization()(x)

    # Level 2 Residual Connections
    for _ in range(2):
        shortcut = x
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(16, kernel_size=(3, 3), activation='relu')(x) 
        x = layers.BatchNormalization()(x)
        branch = layers.Conv2D(16, kernel_size=(1, 1))(x)
        x = layers.add([shortcut, x, branch])

    # Level 3 Global Branch
    residual_features = layers.Conv2D(16, kernel_size=(1, 1))(x) 
    x = layers.add([residual_features, x])

    # Average Pooling and Classification
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model