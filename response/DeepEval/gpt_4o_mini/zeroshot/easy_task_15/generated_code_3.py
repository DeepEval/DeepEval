import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

def dl_model():
    # Input layer
    inputs = layers.Input(shape=(28, 28, 1))  # MNIST images are 28x28 pixels with 1 color channel

    # Specialized block
    def conv_block(x):
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)  # 3x3 convolution
        x = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(x)  # 1x1 convolution
        x = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(x)  # another 1x1 convolution
        x = layers.AveragePooling2D(pool_size=(2, 2))(x)  # Average pooling
        x = layers.Dropout(0.25)(x)  # Dropout to reduce overfitting
        return x

    # Apply the convolutional blocks twice
    x = conv_block(inputs)
    x = conv_block(x)

    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Flatten layer (optional since GlobalAveragePooling returns a 1D vector)
    # x = layers.Flatten()(x)

    # Fully connected layer
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)  # 10 classes for MNIST

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
if __name__ == "__main__":
    model = dl_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_acc}')