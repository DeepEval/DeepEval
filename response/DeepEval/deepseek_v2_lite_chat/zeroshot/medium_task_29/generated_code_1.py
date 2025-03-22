import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

# Load and preprocess CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define the model architecture
def dl_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D(pool_size=(1, 1), strides=None),  # Adjust strides to match window size
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=None),  # Adjust strides to match window size
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(4, 4), strides=None),  # Adjust strides to match window size
        Flatten(),
        Dense(512, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Create and compile the model
model = dl_model()
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)