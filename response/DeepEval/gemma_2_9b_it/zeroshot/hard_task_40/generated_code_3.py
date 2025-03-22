import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    
    # Input Layer
    inputs = layers.Input(shape=(28, 28, 1))

    # Block 1: Average Pooling and Flattening
    x = layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(inputs)
    x = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(x)
    x = layers.Flatten()(x)

    # Concatenate outputs and reshape
    x = layers.Concatenate()([x, x, x])
    x = layers.Reshape((7*7*10,))(x)

    # Fully connected layer
    x = layers.Dense(units=64, activation='relu')(x)

    # Reshape for Block 2
    x = layers.Reshape((8, 8, 1))(x)

    # Block 2: Multi-scale Feature Extraction
    # Path 1: 1x1 convolution
    path1 = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(x)
    path1 = layers.Dropout(0.2)(path1)

    # Path 2: 1x1, 3x3, 3x3 convolutions
    path2 = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(x)
    path2 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(path2)
    path2 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(path2)
    path2 = layers.Dropout(0.2)(path2)

    # Path 3: 1x1, 3x3 convolution
    path3 = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(x)
    path3 = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(path3)
    path3 = layers.Dropout(0.2)(path3)

    # Path 4: Average pooling, 1x1 convolution
    path4 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    path4 = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(path4)
    path4 = layers.Dropout(0.2)(path4)

    # Concatenate outputs
    x = layers.Concatenate(axis=3)([path1, path2, path3, path4])

    # Classification layers
    x = layers.Flatten()(x)
    x = layers.Dense(units=128, activation='relu')(x)
    outputs = layers.Dense(units=10, activation='softmax')(x)

    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model