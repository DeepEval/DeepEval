import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():

    # Input layer
    input_img = keras.Input(shape=(28, 28, 1))

    # Block 1: Parallel Max Pooling Paths
    path1 = layers.MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(input_img)
    path2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_img)
    path3 = layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(input_img)

    # Flatten and Regularize
    path1 = layers.Flatten()(path1)
    path2 = layers.Flatten()(path2)
    path3 = layers.Flatten()(path3)
    path1 = layers.Dropout(0.2)(path1)
    path2 = layers.Dropout(0.2)(path2)
    path3 = layers.Dropout(0.2)(path3)

    # Concatenate Outputs
    concat_path = layers.concatenate([path1, path2, path3])

    # Fully Connected Layer and Reshape
    x = layers.Dense(256, activation='relu')(concat_path)
    x = layers.Reshape((1, 1, 256))(x)

    # Block 2: Multi-Scale Feature Extraction
    path1 = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    path2 = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    path2 = layers.Conv2D(64, (1, 7), padding='same', activation='relu')(path2)
    path2 = layers.Conv2D(64, (7, 1), padding='same', activation='relu')(path2)
    path3 = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    path3 = layers.Conv2D(64, (7, 1), padding='same', activation='relu')(path3)
    path3 = layers.Conv2D(64, (1, 7), padding='same', activation='relu')(path3)
    path3 = layers.Conv2D(64, (7, 1), padding='same', activation='relu')(path3)
    path4 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    path4 = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(path4)

    # Concatenate and Flatten
    concat_path = layers.concatenate([path1, path2, path3, path4])
    concat_path = layers.Flatten()(concat_path)

    # Final Classification Layers
    x = layers.Dense(256, activation='relu')(concat_path)
    output = layers.Dense(10, activation='softmax')(x)

    # Model Definition
    model = keras.Model(inputs=input_img, outputs=output)

    return model