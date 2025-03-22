import tensorflow as tf
from tensorflow import keras

def dl_model():
    input_tensor = keras.Input(shape=(28, 28, 1))

    # First Block
    x = keras.layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_tensor)
    x = keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = keras.layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Concatenate()([x, x, x])
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Reshape((1, 128))(x)

    # Second Block
    path1 = keras.Sequential([
        keras.layers.Conv2D(32, (1, 1), activation='relu'),
        keras.layers.Dropout(0.2)
    ])(x)
    path2 = keras.Sequential([
        keras.layers.Conv2D(32, (1, 1), activation='relu'),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.Dropout(0.2)
    ])(x)
    path3 = keras.Sequential([
        keras.layers.Conv2D(64, (1, 1), activation='relu'),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.Dropout(0.2)
    ])(x)
    path4 = keras.Sequential([
        keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
        keras.layers.Conv2D(64, (1, 1), activation='relu'),
        keras.layers.Dropout(0.2)
    ])(x)
    
    output = keras.layers.Concatenate(axis=1)([path1, path2, path3, path4])
    output = keras.layers.Flatten()(output)
    output = keras.layers.Dense(10, activation='softmax')(output)

    model = keras.Model(inputs=input_tensor, outputs=output)
    
    return model