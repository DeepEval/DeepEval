import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    
    # Define the input shape
    input_shape = (32, 32, 3)

    # 1x1 Convolutional Layer
    x = layers.Input(shape=input_shape)
    x_1x1 = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(x)

    # 3x3 Convolutional Layer
    x_3x3 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)

    # 5x5 Convolutional Layer
    x_5x5 = layers.Conv2D(filters=128, kernel_size=(5, 5), activation='relu')(x)

    # Max Pooling Layer
    x_pool = layers.MaxPooling2D(pool_size=(3, 3))(x_3x3)

    # Concatenate Feature Maps
    x_concat = layers.concatenate([x_1x1, x_3x3, x_5x5, x_pool])

    # Flatten Layer
    x_flat = layers.Flatten()(x_concat)

    # Fully Connected Layer 1
    x = layers.Dense(units=128, activation='relu')(x_flat)

    # Fully Connected Layer 2
    output = layers.Dense(units=10, activation='softmax')(x)

    # Create the model
    model = tf.keras.Model(inputs=x, outputs=output)

    return model