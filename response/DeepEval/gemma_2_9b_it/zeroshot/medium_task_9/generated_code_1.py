from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_layer = keras.Input(shape=(32, 32, 3))  

    x = layers.Conv2D(16, (3, 3), padding='same')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Branch path
    branch = layers.Conv2D(16, (3, 3), padding='same')(input_layer) 

    # Basic block 1
    x = layers.Conv2D(32, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Add()([x, branch]) 

    # Basic block 2
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Add()([x, branch]) 

    # Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Flatten and fully connected layer
    x = layers.Flatten()(x)
    output_layer = layers.Dense(10, activation='softmax')(x)  

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model