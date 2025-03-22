# Import necessary packages from Keras and other libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def dl_model():
    # Create a new Sequential model
    model = keras.Sequential()

    # Main Path
    # Consecutive 3x3 convolutional layers with ReLU activation
    main_path = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.Conv2D(32, (3, 3), activation='relu')
    ])

    # Max pooling layer
    main_path = keras.Sequential([
        main_path,
        layers.MaxPooling2D((2, 2))
    ])

    # Branch Path
    # Single 5x5 convolutional layer with ReLU activation
    branch_path = keras.Sequential([
        layers.Conv2D(64, (5, 5), activation='relu', input_shape=(32, 32, 3))
    ])

    # Combine both paths
    combined_path = keras.Sequential([
        layers.Concatenate()([main_path.output, branch_path.output]),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # Define the overall model architecture
    model = keras.Model(main_path.input, combined_path.output)

    return model


# Define the model, compile it, and train it
def train_model():
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Define the model
    model = dl_model()

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

    return model