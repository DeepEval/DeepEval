from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.datasets import mnist

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize the pixel values to be between 0 and 1
X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255

# Define the input shape for the model
input_shape = (28, 28, 1)

# Define the first convolutional layer with 32 filters and a kernel size of 3x3
conv1 = Conv2D(32, (3, 3), activation='relu')

# Define the max pooling layer with a pool size of 2x2 and a stride of 2
pool1 = MaxPooling2D((2, 2), strides=2)

# Define the first convolutional block with 64 filters and kernel sizes of 1x1 and 3x3
conv2 = Conv2D(64, (1, 1), activation='relu')
conv3 = Conv2D(64, (3, 3), activation='relu')

# Define the second convolutional block with 64 filters and kernel sizes of 1x1 and 3x3
conv4 = Conv2D(64, (1, 1), activation='relu')
conv5 = Conv2D(64, (3, 3), activation='relu')

# Define the third convolutional block with 64 filters and a kernel size of 5x5
conv6 = Conv2D(64, (5, 5), activation='relu')

# Define the third max pooling layer with a pool size of 2x2
pool2 = MaxPooling2D((2, 2), strides=2)

# Flatten the feature maps from the previous layer to create a 1D array
flatten = Flatten()

# Define three fully connected layers with 128, 64, and 10 neurons, respectively
dense1 = Dense(128, activation='relu')
dense2 = Dense(64, activation='relu')
dense3 = Dense(10, activation='softmax')

# Create a Sequential model and add the layers to it
model = Sequential()
model.add(conv1)
model.add(pool1)
model.add(conv2)
model.add(conv3)
model.add(pool1)
model.add(conv4)
model.add(conv5)
model.add(pool2)
model.add(flatten)
model.add(dense1)
model.add(dense2)
model.add(dense3)

# Compile the model with a loss function and an optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model on the MNIST dataset
model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test))