from tensorflow import keras
from tensorflow.keras import layers

ENCODING_DIM = 64

def method():
  model = keras.Sequential(
    [
      layers.Flatten(input_shape=(28 * 28,)),  # Flatten the 28x28 MNIST image
      layers.Dense(ENCODING_DIM, activation='relu')  # Dense layer to the desired encoding dimension
    ]
  )
  return model

# Call the method to generate the model
model = method()

# You can now inspect the model structure:
print(model.summary())