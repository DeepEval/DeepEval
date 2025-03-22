import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))

    # First block: parallel branches
    branch1 = layers.Conv2D(filters=32, kernel_size=(1, 1))(inputs)
    branch2 = layers.Conv2D(filters=32, kernel_size=(3, 3))(inputs)
    branch3 = layers.Conv2D(filters=32, kernel_size=(5, 5))(inputs)
    branch4 = layers.MaxPooling2D(pool_size=(3, 3))(inputs)

    # Concatenate outputs from branches
    x = layers.Concatenate()([branch1, branch2, branch3, branch4])

    # Second block: global average pooling and dense layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    
    # Reshape and element-wise multiplication
    weights = layers.Dense(32, activation='linear')(x)  
    weights = layers.Reshape((32, 32, 3))(weights)
    output = inputs * weights 

    # Final fully connected layer
    outputs = layers.Dense(10, activation='softmax')(output)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model