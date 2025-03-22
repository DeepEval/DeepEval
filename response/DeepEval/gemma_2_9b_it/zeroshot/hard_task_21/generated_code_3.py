import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))

    # Main Path
    x = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=2))(inputs)
    
    # 1x1 branch
    x_1 = layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(x[0])
    
    # 3x3 branch
    x_2 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x[1])

    # 5x5 branch
    x_3 = layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same')(x[2])

    x = layers.concatenate([x_1, x_2, x_3], axis=2)

    # Branch Path
    x_branch = layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(inputs)

    # Add outputs
    x = layers.Add()([x, x_branch])

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Fully Connected Layers
    x = layers.Dense(units=128, activation='relu')(x)
    outputs = layers.Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model