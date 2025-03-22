import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

def method():
    # 1. Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # 2. Preprocess the data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

    # 3. Define the model
    model = keras.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(10, activation='softmax'),
        ]
    )

    # 4. Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 5. Train the model
    model.fit(x_train, y_train, epochs=5)

    # 6. Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f'Test loss: {loss}')
    print(f'Test accuracy: {accuracy}')

    return accuracy

# Call the method to validate
accuracy = method()
print(f"Final accuracy: {accuracy}")