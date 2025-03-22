from tensorflow.keras import Model, Input, layers, Sequential
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the model
def dl_model():
  # Input layer
  inputs = Input(shape=(32, 32, 3))

  # Path 1
  path_1 = layers.Conv2D(64, (1, 1), padding='same')(inputs)

  # Path 2
  path_2 = layers.AveragePooling2D()(inputs)
  path_2 = layers.Conv2D(64, (1, 1), padding='same')(path_2)

  # Path 3
  path_3 = layers.Conv2D(64, (1, 1), padding='same')(inputs)
  path_3 = layers.Conv2D(64, (1, 3), padding='same')(path_3)
  path_3 = layers.Conv2D(64, (3, 1), padding='same')(path_3)
  path_3 = layers.Concatenate()([path_3, path_3, path_3])

  # Path 4
  path_4 = layers.Conv2D(64, (1, 1), padding='same')(inputs)
  path_4 = layers.Conv2D(64, (3, 3), padding='same')(path_4)
  path_4 = layers.Conv2D(64, (1, 3), padding='same')(path_4)
  path_4 = layers.Conv2D(64, (3, 1), padding='same')(path_4)
  path_4 = layers.Concatenate()([path_4, path_4, path_4])

  # Concatenate all paths
  output = layers.Concatenate()([path_1, path_2, path_3, path_4])

  # Fully connected layer
  output = layers.Flatten()(output)
  output = layers.Dense(10, activation='softmax')(output)

  # Create the model
  model = Model(inputs=inputs, outputs=output)

  # Compile the model
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  # Train the model
  model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), callbacks=[EarlyStopping(), ReduceLROnPlateau()])

  return model

# Print the model summary
model = dl_model()
model.summary()