import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    inputs = layers.Input(shape=(28, 28, 1))

    # Main Path
    # First block
    x = layers.SeparableConv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    # Second block
    x = layers.SeparableConv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Branch Path
    # 1x1 convolution to match dimensions
    branch = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(inputs)

    # Combine both paths
    combined = layers.add([x, branch])

    # Flatten and fully connected layer
    combined = layers.Flatten()(combined)
    outputs = layers.Dense(10, activation='softmax')(combined)

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example of how to use the model
if __name__ == "__main__":
    model = dl_model()
    model.summary()

    # Load and preprocess MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255.0
    x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_accuracy:.4f}')