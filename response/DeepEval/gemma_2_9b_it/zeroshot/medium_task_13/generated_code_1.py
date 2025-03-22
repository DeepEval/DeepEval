from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    model = keras.Sequential()

    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3)) 

    # Convolutional Layer 1
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    
    # Concatenate outputs
    x = layers.concatenate([input_layer, conv1], axis=-1)

    # Convolutional Layer 2
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    
    # Concatenate outputs
    x = layers.concatenate([x, conv2], axis=-1)

    # Convolutional Layer 3
    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    
    # Flatten for fully connected layers
    x = layers.Flatten()(conv3)

    # Fully Connected Layer 1
    dense1 = layers.Dense(128, activation='relu')(x)

    # Output Layer
    output_layer = layers.Dense(10, activation='softmax')(dense1)

    # Create model
    model = keras.Model(inputs=input_layer, outputs=output_layer) 

    return model