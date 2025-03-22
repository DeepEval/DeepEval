import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_tensor = keras.Input(shape=(32, 32, 3))

    # First Block
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=2))(input_tensor)
    
    # Group 1
    group1 = layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(x[0])
    # Group 2
    group2 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x[1])
    # Group 3
    group3 = layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(x[2])
    
    x = layers.Concatenate(axis=2)([group1, group2, group3])

    # Second Block
    branch1 = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(x)
    branch2 = layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(x)
    branch2 = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(branch2)
    branch2 = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(branch2)
    branch3 = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Concatenate(axis=3)([branch1, branch2, branch3])

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Output Layer
    output = layers.Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_tensor, outputs=output)
    return model