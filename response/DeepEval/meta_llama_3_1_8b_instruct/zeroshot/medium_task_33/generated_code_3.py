# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize pixel values between 0 and 1
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

def dl_model():
    # Define the input layer
    inputs = keras.Input(shape=(32, 32, 3))
    
    # Split the input into three channel groups
    channel_groups = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(inputs)
    
    # Define the first group
    group1 = layers.Conv2D(32, 1, activation='relu')(channel_groups[0])
    group1 = layers.Conv2D(32, 3, activation='relu')(group1)
    group1 = layers.Conv2D(64, 5, activation='relu')(group1)
    
    # Define the second group
    group2 = layers.Conv2D(32, 1, activation='relu')(channel_groups[1])
    group2 = layers.Conv2D(32, 3, activation='relu')(group2)
    group2 = layers.Conv2D(64, 5, activation='relu')(group2)
    
    # Define the third group
    group3 = layers.Conv2D(32, 1, activation='relu')(channel_groups[2])
    group3 = layers.Conv2D(32, 3, activation='relu')(group3)
    group3 = layers.Conv2D(64, 5, activation='relu')(group3)
    
    # Concatenate the outputs from the three groups
    outputs = layers.Concatenate()([group1, group2, group3])
    
    # Define the dense layers
    outputs = layers.Flatten()(outputs)
    outputs = layers.Dense(128, activation='relu')(outputs)
    outputs = layers.Dense(64, activation='relu')(outputs)
    outputs = layers.Dense(10, activation='softmax')(outputs)
    
    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model

# Compile the model
model = dl_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))