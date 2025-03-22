import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard


def method():
    # Define the model architecture
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Define the TensorBoard callback
    log_dir = 'logs/tensorboard'
    tensorboard_callback = TensorBoard(log_dir=log_dir)

    # Train the model
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 784).astype('float32') / 255.0
    model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), callbacks=[tensorboard_callback])

    # Return the output
    return 'Training complete. You can now visualize the session using TensorBoard at the log directory:'+ log_dir

# Call the method for validation
output = method()
print(output)