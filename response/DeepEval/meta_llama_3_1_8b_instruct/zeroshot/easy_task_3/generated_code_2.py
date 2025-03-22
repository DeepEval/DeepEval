# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the deep learning model function
def dl_model():
    # Define the input shape (28x28 pixels for MNIST dataset)
    input_shape = (28, 28, 1)
    
    # Define the model architecture
    model = keras.Sequential([
        # Convolution block 1: convolution, convolution, max pooling
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Convolution block 2: convolution, convolution, convolution, max pooling
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten the feature maps
        layers.Flatten(),
        
        # Fully connected layers for classification
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')  # Output layer with 10 classes (digits 0-9)
    ])
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Test the model
if __name__ == "__main__":
    # Load MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
    
    # Reshape and normalize the input data
    train_images = train_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    test_images = test_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    # Build and train the model
    model = dl_model()
    model.fit(train_images, train_labels, epochs=5, batch_size=128, validation_data=(test_images, test_labels))
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f'Test accuracy: {test_acc:.2f}')