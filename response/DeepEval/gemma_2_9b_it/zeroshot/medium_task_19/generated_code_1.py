import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    
    # Define the four branches
    branch1 = tf.keras.Sequential([
        layers.Conv2D(32, (1, 1), activation='relu', input_shape=(32, 32, 3)) 
    ])

    branch2 = tf.keras.Sequential([
        layers.Conv2D(32, (1, 1), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu')
    ])

    branch3 = tf.keras.Sequential([
        layers.Conv2D(32, (1, 1), activation='relu'),
        layers.Conv2D(64, (5, 5), activation='relu')
    ])

    branch4 = tf.keras.Sequential([
        layers.MaxPooling2D((3, 3), strides=(2, 2)),
        layers.Conv2D(64, (1, 1), activation='relu')
    ])

    # Combine the outputs of the branches
    combined = layers.concatenate([branch1.output, branch2.output, branch3.output, branch4.output])

    # Flatten the combined features
    x = layers.Flatten()(combined)

    # Add two fully connected layers for classification
    x = layers.Dense(128, activation='relu')(x)
    output = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = models.Model(inputs=branch1.input, outputs=output)

    return model