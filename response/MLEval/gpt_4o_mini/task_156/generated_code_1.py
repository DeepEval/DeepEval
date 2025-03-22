import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Flatten
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

def method():
    # Generate some sample data
    # For the sake of example, let's create dummy data
    num_samples = 1000
    max_sequence_length = 10
    vocab_size = 50  # Vocabulary size for the embedding layer
    embedding_dim = 8  # Dimension of embedding vector
    
    # Randomly generate sequences of integers
    X = np.random.randint(1, vocab_size, size=(num_samples, max_sequence_length))
    y = np.random.randint(0, 2, size=(num_samples,))  # Binary classification

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Pad sequences (if needed)
    X_train = pad_sequences(X_train, maxlen=max_sequence_length)
    X_test = pad_sequences(X_test, maxlen=max_sequence_length)

    # Build the LSTM model
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length))
    model.add(LSTM(50))  # LSTM layer with 50 units
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    return model, loss, accuracy

# Call the method for validation
output = method()