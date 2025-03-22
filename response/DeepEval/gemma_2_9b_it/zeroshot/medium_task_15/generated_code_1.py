import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    inputs = layers.Input(shape=(32, 32, 3))

    # Convolutional layer
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Fully connected layers
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)

    # Reshape and concatenate
    x = layers.Reshape((x.shape[1], 32, 32, 32))(x)  
    x = layers.Concatenate()([x, inputs])

    # 1x1 Convolution and Average Pooling
    x = layers.Conv2D(16, (1, 1), activation='relu')(x)
    x = layers.AveragePooling2D((2, 2))(x)

    # Output layer
    outputs = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model