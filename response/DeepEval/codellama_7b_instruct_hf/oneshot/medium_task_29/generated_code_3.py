import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Model

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Convert class labels to binary vectors
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define the input shape and number of classes
input_shape = (32, 32, 3)
num_classes = 10

# Define the model
model = Sequential()

# Convolutional layer 1
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))

# Convolutional layer 2
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Convolutional layer 3
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Flatten the output from the 3D feature maps to a 1D vector
model.add(Flatten())

# Dense layer 1
model.add(Dense(128, activation='relu'))

# Dense layer 2
model.add(Dense(64, activation='relu'))

# Dense layer 3 (output layer)
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model on the CIFAR-10 dataset
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model on the test dataset
loss, accuracy = model.evaluate(x_test, y_test)
print('Test accuracy:', accuracy)