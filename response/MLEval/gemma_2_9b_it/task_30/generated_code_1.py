import tensorflow as tf

def method():
    # Define the model architecture
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),  # Input layer
        tf.keras.layers.Dense(32, activation='relu'),  # Hidden layer
        tf.keras.layers.Dense(10, activation='softmax')  # Output layer
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Call the function and print the returned model
dnn_model = method()
print(dnn_model.summary())