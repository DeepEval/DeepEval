import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))

    # Main Path
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=1))(inputs)  
    
    # 1x1 Convolutional Group
    branch1 = layers.Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(x[0])
    
    # 3x3 Convolutional Group
    branch2 = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x[1])

    # 5x5 Convolutional Group
    branch3 = layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(x[2])
    
    # Concatenate outputs from main path branches
    x = layers.concatenate([branch1, branch2, branch3], axis=1)

    # Branch Path
    branch = layers.Conv2D(filters=128, kernel_size=(1, 1), activation='relu')(inputs)

    # Combine Main and Branch Paths
    x = layers.Add()([x, branch])

    # Fully Connected Layers
    x = layers.Flatten()(x)
    x = layers.Dense(units=128, activation='relu')(x)
    outputs = layers.Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model