from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def method():
  # Define the model
  model = Sequential()
  model.add(Dense(64, activation='relu', input_shape=(10,)))
  model.add(Dense(10, activation='softmax'))

  # Compile the model
  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  # Fit the model
  history = model.fit(x_train, y_train, epochs=10, batch_size=32)

  # Evaluate the model
  loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
  print(f'Loss: {loss}, Accuracy: {accuracy}')

  # Return the history object
  return history 

# Call the method and print the output
history = method()

# Print history
print(history.history)