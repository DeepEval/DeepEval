from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

def dl_model():

  # Load the CIFAR-10 dataset
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()

  # Preprocess the data
  x_train = x_train.astype('float32') / 255.0
  x_test = x_test.astype('float32') / 255.0

  # Define the model architecture
  inputs = layers.Input(shape=(32, 32, 3))

  # Convolutional layers with channel concatenation
  conv1 = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
  conv2 = layers.Conv2D(64, (3, 3), activation='relu')(conv1)
  conv3 = layers.Conv2D(128, (3, 3), activation='relu')(conv2)

  # Flatten and fully connected layers
  x = layers.concatenate([conv1, conv2, conv3])
  x = layers.Flatten()(x)
  x = layers.Dense(256, activation='relu')(x)
  outputs = layers.Dense(10, activation='softmax')(x)

  # Create the model
  model = models.Model(inputs=inputs, outputs=outputs)

  return model