import tensorflow as tf
import numpy as np
import datetime

def method():
    # Create a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Generate dummy data for the example
    x_train = np.random.rand(60000, 784)  # 60,000 samples, 784 features
    y_train = np.random.randint(0, 10, size=(60000,))  # 60,000 labels (0-9)

    # Set up TensorBoard callback
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Train the model while logging to TensorBoard
    model.fit(x_train, y_train, epochs=5, callbacks=[tensorboard_callback])

    return log_dir

# Call the method to validate
output = method()
print("TensorBoard logs can be found at:", output)