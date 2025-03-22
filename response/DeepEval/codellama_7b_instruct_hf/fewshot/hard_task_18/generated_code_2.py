import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)
    
    # Define the first block
    first_block = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2), strides=2),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2), strides=2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu')
    ])
    
    # Define the second block
    second_block = tf.keras.Sequential([
        layers.GlobalAveragePooling2D(),
        layers.Dense(32, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Flatten()
    ])
    
    # Define the model
    model = tf.keras.Sequential([
        first_block,
        layers.Add(),
        second_block,
        layers.Flatten(),
        layers.Dense(10, activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    return model