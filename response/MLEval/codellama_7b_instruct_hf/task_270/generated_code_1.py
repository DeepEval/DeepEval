import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Generate some random data for training
X = np.random.rand(100, 10)
y = np.random.rand(100)

# Set up the model
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(10,)))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10, batch_size=10)

# Generate some random data for testing
test_X = np.random.rand(10, 10)
test_y = np.random.rand(10)

# Make predictions
predictions = model.predict(test_X)

# Evaluate the model
loss, accuracy = model.evaluate(test_X, test_y)
print('Loss: {}, Accuracy: {}'.format(loss, accuracy))