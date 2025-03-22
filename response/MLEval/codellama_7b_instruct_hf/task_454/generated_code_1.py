import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Define the input and output shapes
input_shape = (4,)
output_shape = (3,)

# Create the sequential model
model = Sequential()

# Add the first layer (dense) with the appropriate number of neurons and activation function
model.add(Dense(64, activation='relu', input_shape=input_shape))

# Add the second layer (dense) with the appropriate number of neurons and activation function
model.add(Dense(32, activation='relu'))

# Add the output layer (dense) with the appropriate number of neurons and activation function
model.add(Dense(output_shape[1], activation='softmax'))

# Compile the model with the appropriate loss function, optimizer, and metrics
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model on a dataset
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Use the model to make predictions on new data
predictions = model.predict(new_data)

# Evaluate the performance of the model on the test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)