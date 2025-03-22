# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def dl_model():
    # Define the input layer
    inputs = keras.Input(shape=(32, 32, 3))

    # Block 1
    def block_1(x):
        group1 = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(x)
        group1 = layers.Conv2D(6, (1, 1), activation='relu')(group1[0])
        group1 = layers.Conv2D(6, (3, 3), activation='relu')(group1)
        group1 = layers.Conv2D(6, (1, 1), activation='relu')(group1)
        
        group2 = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(x)
        group2 = layers.Conv2D(6, (1, 1), activation='relu')(group2[0])
        group2 = layers.Conv2D(6, (3, 3), activation='relu')(group2)
        group2 = layers.Conv2D(6, (1, 1), activation='relu')(group2)
        
        group3 = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(x)
        group3 = layers.Conv2D(6, (1, 1), activation='relu')(group3[0])
        group3 = layers.Conv2D(6, (3, 3), activation='relu')(group3)
        group3 = layers.Conv2D(6, (1, 1), activation='relu')(group3)
        
        x = layers.Concatenate()([group1, group2, group3])
        return x
    
    x = block_1(inputs)

    # Transition Convolution
    x = layers.Conv2D(12, (1, 1), activation='relu')(x)
    x = layers.AveragePooling2D((2, 2))(x)

    # Block 2
    def block_2(x):
        x = layers.GlobalMaxPooling2D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(128, activation='relu')(x)
        weights = layers.Dense(12, activation='sigmoid')(x)
        weights = layers.Reshape((1, 1, 12))(weights)
        x = x * weights
        return x
    
    branch = layers.Lambda(lambda x: x)(x)
    x = block_2(x)
    x = layers.Add()([x, branch])
    x = layers.Dense(10, activation='softmax')(x)
    
    # Define the model
    model = keras.Model(inputs=inputs, outputs=x)
    return model

# Define the model
model = dl_model()
model.summary()