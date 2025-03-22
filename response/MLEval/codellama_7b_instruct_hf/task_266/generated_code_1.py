import tensorflow as tf
from tensorflow.keras.layers import Dense

def method():
    # Define the hyperparameters for the model
    batch_size = 32
    num_epochs = 10

    # Define the model architecture
    model = tf.keras.Sequential([
        Dense(64, activation='relu', input_shape=(784,)),
        Dense(10, activation='softmax')
    ])

    # Compile the model with the appropriate loss function and optimizer
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model using the training data
    model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size)

    # Evaluate the model using the validation data
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test loss: {loss:.3f}, Test accuracy: {accuracy:.3f}')

    return model

# Call the generated 'method()' for validation
method()