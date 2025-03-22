import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def method():
    # Create a sequential model
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(784,)),  # First dense layer
        layers.Dropout(0.2),  # Dropout layer to reduce overfitting
        layers.Dense(32, activation='relu'),  # Second dense layer
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')  # Output layer with 10 units for 10 classes
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Generate dummy input data
    input_data = tf.random.normal(shape=(100, 784))
    input_labels = tf.random.uniform(shape=(100,), minval=0, maxval=10, dtype=tf.int32)

    # Train the model
    model.fit(input_data, input_labels, epochs=5)

    # Make predictions
    predictions = model.predict(input_data)

    # Return the predictions
    return predictions

# Call the method and print the results
predictions = method()
print(predictions)