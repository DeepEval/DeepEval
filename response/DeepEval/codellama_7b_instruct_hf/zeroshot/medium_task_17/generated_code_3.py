from keras.applications import VGG16
from keras.layers import Input, Dense, Flatten, Reshape, Permute
from keras.models import Model

# Load the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Define the input shape
input_shape = (32, 32, 3)

# Define the model architecture
model = Sequential()
model.add(Reshape((3, 32, 32), input_shape=input_shape))
model.add(Permute((2, 1, 3), input_shape=input_shape))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))