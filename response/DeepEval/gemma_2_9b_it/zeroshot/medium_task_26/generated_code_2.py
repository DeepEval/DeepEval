import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    input_shape = (32, 32, 64)
    
    # Input layer
    inputs = layers.Input(shape=input_shape)

    # 1x1 Convolutional Layer
    x = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(inputs)

    # Parallel Convolutional Layers
    conv1x1 = layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(x)
    conv3x3 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)

    # Concatenate the outputs
    x = layers.concatenate([conv1x1, conv3x3], axis=-1)

    # Flatten the output
    x = layers.Flatten()(x)

    # Fully Connected Layers
    x = layers.Dense(units=128, activation='relu')(x)
    outputs = layers.Dense(units=10, activation='softmax')(x)  # Adjust units for your number of classes

    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model