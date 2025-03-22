import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape and normalize data
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255

# One-hot encode the labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define the model
def dl_model():
    model = Sequential([
        # Reduce dimensionality with a 1x1 convolution
        Conv2D(32, (1, 1), activation='relu', input_shape=(28, 28, 1)),
        # Extract features with a 3x3 convolutional layer
        Conv2D(64, (3, 3), activation='relu'),
        # Restore dimensionality with another 1x1 convolution
        Conv2D(32, (1, 1), activation='relu'),
        # Flatten the output
        Flatten(),
        # Feed into a fully connected layer for classification
        Dense(10, activation='softmax')
    ])
    
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Build the model
model = dl_model()
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f'Test Accuracy: {accuracy*100:.2f}%')

# Save the model
model.save('mnist_classifier.h5')