import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input Layer
    inputs = tf.keras.Input(shape=(28, 28, 1)) 

    # Block 1: Average Pooling and Flattening
    x = layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(inputs)
    x = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(x)
    x = layers.Flatten()(x)
    x = layers.concatenate([x, x, x], axis=-1)  # Concatenate the outputs
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Reshape((1, 128))(x) 

    # Block 2: Multi-Scale Feature Extraction
    # Path 1
    path1 = layers.Conv2D(32, (1, 1))(x)
    path1 = layers.Dropout(0.2)(path1)

    # Path 2
    path2 = layers.Conv2D(32, (1, 1))(x)
    path2 = layers.Conv2D(64, (3, 3))(path2)
    path2 = layers.Conv2D(64, (3, 3))(path2)
    path2 = layers.Dropout(0.2)(path2)

    # Path 3
    path3 = layers.Conv2D(64, (1, 1))(x)
    path3 = layers.Conv2D(64, (3, 3))(path3)
    path3 = layers.Dropout(0.2)(path3)

    # Path 4
    path4 = layers.AveragePooling2D(pool_size=(2, 2))(x)
    path4 = layers.Conv2D(32, (1, 1))(path4)
    path4 = layers.Dropout(0.2)(path4)

    # Concatenate outputs
    output_concat = layers.concatenate([path1, path2, path3, path4], axis=-1)

    # Classification Layers
    output_concat = layers.Flatten()(output_concat)
    output = layers.Dense(10, activation='softmax')(output_concat) 

    model = models.Model(inputs=inputs, outputs=output)

    return model