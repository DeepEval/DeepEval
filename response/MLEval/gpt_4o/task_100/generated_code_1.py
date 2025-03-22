# Import necessary packages
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
import numpy as np

def method():
    # Load the CIFAR-10 dataset
    # (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # # Normalize pixel values to be between 0 and 1
    # train_images, test_images = train_images / 255.0, test_images / 255.0

    # # One-hot encode the labels
    # train_labels = to_categorical(train_labels, 10)
    # test_labels = to_categorical(test_labels, 10)

    # Define the CNN model
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Flatten the output and add dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    # model.fit(train_images, train_labels, epochs=10, 
    #           validation_data=(test_images, test_labels))

    # # Evaluate the model on the test set
    # test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    # # Return the test accuracy
    # output = test_acc
    # return output

# Call the method for validation
# accuracy = method()
# print(f"Test accuracy: {accuracy:.2f}")
method()