import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_tensor = keras.Input(shape=(32, 32, 3))

    # Main path
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=2))(input_tensor)
    
    # 1x1 branch
    x1 = layers.Conv2D(64, (1, 1), activation='relu', padding='same')(x[0])
    
    # 3x3 branch
    x2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x[1])
    
    # 5x5 branch
    x3 = layers.Conv2D(64, (5, 5), activation='relu', padding='same')(x[2])

    # Concatenate outputs of the three branches
    x = layers.concatenate([x1, x2, x3], axis=3)

    # Branch path
    x_branch = layers.Conv2D(64, (1, 1), activation='relu', padding='same')(input_tensor)

    # Add outputs from main and branch paths
    x = layers.add([x, x_branch])

    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Fully connected layers
    x = layers.Dense(128, activation='relu')(x)
    output_tensor = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=input_tensor, outputs=output_tensor)
    return model