import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def method():
  # Define the CNN model using RGB input
  model = Sequential()
  model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3))) # Input shape for RGB images (height, width, channels)
  model.add(MaxPooling2D((2, 2)))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D((2, 2)))
  model.add(Flatten())
  model.add(Dense(10, activation='softmax')) # Adjust number of output units based on your task

  # Compile the model
  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  return model

# Call the method to create the model
model = method()

# Now you can use this model to train or evaluate your RGB image data
# For example:
# model.fit(x_train, y_train, epochs=10)
# loss, accuracy = model.evaluate(x_test, y_test)