from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 64))

    # 1x1 Convolutional Layer for channel compression
    x = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(inputs)

    # Parallel Convolutional Layers
    conv1x1 = layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(x)
    conv3x3 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)

    # Concatenate the outputs of the parallel convolutions
    x = layers.concatenate([conv1x1, conv3x3], axis=-1)

    # Flatten the output feature map
    x = layers.Flatten()(x)

    # Fully Connected Layers
    x = layers.Dense(units=128, activation='relu')(x)
    outputs = layers.Dense(units=10, activation='softmax')(x) 

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model