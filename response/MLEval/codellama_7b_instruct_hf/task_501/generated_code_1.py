from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy
import numpy as np

# Generate some random data
X = np.random.rand(1000, 20)
y = np.random.randint(0, 2, size=(1000, 1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the data into categorical variables
y_train_categorical = to_categorical(y_train)
y_test_categorical = to_categorical(y_test)

# Define the model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(20,)))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_categorical, epochs=10, batch_size=32, validation_data=(X_test, y_test_categorical))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test_categorical)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

# Make predictions
predictions = model.predict(X_test)

# Convert the predictions to class labels
predictions_classes = np.argmax(predictions, axis=1)

# Print the predictions
print(predictions_classes)