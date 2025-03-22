# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape input data
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Normalize pixel values
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define the deep learning model
def dl_model():
    # Input layer with shape (28, 28, 1)
    inputs = keras.Input(shape=(28, 28, 1))
    
    # 1x1 convolutional layer to reduce dimensionality
    x = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    
    # 3x3 convolutional layer to extract features
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    
    # 1x1 convolutional layer to restore dimensionality
    x = layers.Conv2D(32, (1, 1), activation='relu')(x)
    
    # Flatten the output
    x = layers.Flatten()(x)
    
    # Fully connected layer with 10 neurons for classification
    outputs = layers.Dense(10, activation='softmax')(x)
    
    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, 
          validation_data=(X_val, y_val), 
          batch_size=128)

# Evaluate the model
model.evaluate(X_test, y_test)