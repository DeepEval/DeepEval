import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

def dl_model():
    
    model = models.Sequential()

    # Input Layer
    model.add(layers.Input(shape=(32, 32, 3)))

    # First Block
    model.add(layers.Lambda(lambda x: tf.split(x, 3, axis=-1)))  
    model.add(layers.Conv2D(32, (1, 1), activation='relu', padding='same', name='conv1x1'))
    model.add(layers.BatchNormalization()) 
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv3x3'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (5, 5), activation='relu', padding='same', name='conv5x5'))
    model.add(layers.BatchNormalization()) 
    model.add(layers.Concatenate(axis=-1)) 

    # Second Block
    x = model.output
    branch1 = layers.Conv2D(64, (1, 1), activation='relu', padding='same')(x)
    branch1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(branch1)
    branch2 = layers.Conv2D(64, (1, 1), activation='relu', padding='same')(x)
    branch2 = layers.Conv2D(64, (1, 7), activation='relu', padding='same')(branch2)
    branch2 = layers.Conv2D(64, (7, 1), activation='relu', padding='same')(branch2)
    branch2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(branch2)
    branch3 = layers.AveragePooling2D((2, 2))(x)
    branch3 = layers.Conv2D(64, (1, 1), activation='relu', padding='same')(branch3)

    # Concatenate Branches
    x = layers.Concatenate()([branch1, branch2, branch3])

    # Fully Connected Layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Create the Model
    model = models.Model(inputs=model.input, outputs=outputs)

    return model