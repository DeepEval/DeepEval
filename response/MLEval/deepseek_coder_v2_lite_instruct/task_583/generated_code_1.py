import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def method():
    # Define the model
    model = Sequential()
    
    # Add layers to the model
    model.add(Dense(128, activation='relu', input_shape=(784,)))  # Input layer with 784 features (28x28 images)
    model.add(Dropout(0.2))  # Dropout layer to prevent overfitting
    model.add(Dense(64, activation='relu'))  # Hidden layer
    model.add(Dropout(0.2))  # Dropout layer to prevent overfitting
    model.add(Dense(10, activation='softmax'))  # Output layer with 10 classes (for MNIST dataset)
    
    # Compile the model
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # For demonstration purposes, let's assume we are using the MNIST dataset
    # Load the MNIST dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize the data
    
    # Reshape the data to fit the input layer of the model
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)
    
    # Train the model
    model.fit(x_train, y_train, epochs=5)
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    
    # Return the final output (accuracy in this case)
    return test_acc

# Call the method for validation
output = method()
print(f"Model accuracy: {output}")