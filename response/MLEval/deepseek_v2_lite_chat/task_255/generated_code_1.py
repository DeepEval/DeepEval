import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Assuming the input is a tensor of shape (batch_size, num_features)
def method(input_tensor):
    """
    Apply softmax to the final fully connected layer.

    Parameters:
    input_tensor (Tensor): The input tensor to apply softmax to. It should have shape (batch_size, num_features).

    Returns:
    Tensor: The softmax-normalized tensor.
    """
    # Create a Sequential model to ensure the layers are applied in sequence
    model = Sequential()
    model.add(Dense(num_units, activation='linear', input_shape=(num_features,)))
    model.add(Dense(num_classes))  # Assuming num_classes for the final layer

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    # Fit the model
    model.fit(input_tensor, labels, epochs=epochs, batch_size=batch_size)

    # Predict
    output = model.predict(input_tensor)

    # Apply softmax
    output = tf.nn.softmax(output, axis=1)

    return output

# Example usage
input_tensor = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
labels = tf.constant([[0., 0., 1.], [1., 0., 0.]])  # Binary classification labels

softmax_output = method(input_tensor)
print("Softmax Output:\n", softmax_output)