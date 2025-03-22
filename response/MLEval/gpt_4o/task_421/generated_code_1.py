import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.utils import to_categorical

def method():
    # Assuming X_train_indices and Y_train_oh are defined elsewhere
    # For this example, let's create dummy data
    num_samples = 1000
    max_sequence_length = 100
    num_classes = 5

    # Dummy input data
    X_train_indices = np.random.randint(0, 10000, (num_samples, max_sequence_length))
    # Dummy output data as one-hot encoded
    Y_train_oh = to_categorical(np.random.randint(0, num_classes, num_samples), num_classes=num_classes)

    # Define a simple Keras model
    model = Sequential([
        Embedding(input_dim=10000, output_dim=128, input_length=max_sequence_length),
        LSTM(units=128, return_sequences=False),
        Dense(units=num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Fit the model
    model.fit(X_train_indices, Y_train_oh, epochs=50, batch_size=32, verbose=1)

    # There is no specific 'output' to return for this task as we are just fitting the model
    output = "Model training completed"
    
    return output

# Call the method for validation
result = method()
print(result)