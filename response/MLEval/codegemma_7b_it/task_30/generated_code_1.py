import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def method():
  # Define the DNN architecture
  model = keras.Sequential([
    layers.Input(shape=(100,)),  # Input layer with 100 neurons
    layers.Dense(512, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # Output layer with 10 neurons (assuming 10 classes)
  ])

  # Compile the model
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  # Return the model
  return model

# Call the method to create the DNN
dnn = method()

# Print the DNN architecture
print(dnn.summary())