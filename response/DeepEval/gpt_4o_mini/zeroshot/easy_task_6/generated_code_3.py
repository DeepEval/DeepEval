import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(28, 28, 1))

    # Main path
    # First convolutional layer (increasing feature map width)
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    
    # Second convolutional layer (restoring number of channels)
    x = layers.Conv2D(filters=1, kernel_size=(3, 3), padding='same', activation='relu')(x)

    # Branch path (direct connection to input)
    branch = input_layer

    # Combine paths via addition
    combined = layers.add([x, branch])

    # Flatten the combined output
    flattened = layers.Flatten()(combined)

    # Fully connected layer
    output = layers.Dense(units=10, activation='softmax')(flattened)

    # Create the model
    model = models.Model(inputs=input_layer, outputs=output)

    return model

# Example usage
if __name__ == "__main__":
    model = dl_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Load and preprocess MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255.0
    x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255.0
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)
    
    # Train the model
    model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))