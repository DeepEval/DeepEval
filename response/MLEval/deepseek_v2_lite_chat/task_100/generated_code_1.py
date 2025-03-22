import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


def method():
    # Load the CIFAR-10 dataset
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Convert labelsto binary class matrices
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    # Define the model architecture
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Define early stopping to stop training when validation loss stops decreasing
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    # Train the model
    history = model.fit(train_images, train_labels, epochs=100,
                        validation_data=(test_images, test_labels),
                        callbacks=[early_stopping])

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)

    # Plot training results
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()

    # Predict classes for the test set
    predictions = model.predict(test_images)
    print(predictions[0])

    # Convert prediction classes to one-hot encoding to match test_labels
    predictions = np.argmax(predictions, axis=1)
    print(predictions[:5])  # Print the first 5 predictions
    test_labels = np.argmax(test_labels, axis=1)
    print(test_labels[:5])  # Print the first 5 true labels

    # Compute and print the confusion matrix
    cm = tf.math.confusion_matrix(labels=test_labels, predictions=predictions)
    print(cm)

    # Return the final 'output' as the model's accuracy on the test set
    return test_acc

# Call the method for validation
output = method()