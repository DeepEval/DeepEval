import tensorflow as tf
import numpy as np

def method():
    # Load and prepare the data
    X = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * X)

    # Create a simple linear regression model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=1)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(X, y, epochs=100)

    # Predict the output for a new input
    new_input = np.array([[0.5]])
    prediction = model.predict(new_input)

    # Return the predicted output
    return prediction

# Call the method for validation
output = method()
print(output)