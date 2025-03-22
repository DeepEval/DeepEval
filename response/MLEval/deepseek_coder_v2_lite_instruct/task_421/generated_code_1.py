import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from sklearn.preprocessing import OneHotEncoder

def method():
    # Assuming X_train_indices and Y_train_oh are already defined and loaded with appropriate data
    
    # Define the model
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length))
    model.add(LSTM(units=64, return_sequences=True))
    model.add(LSTM(units=64))
    model.add(Dense(units=num_classes, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Fit the model
    history = model.fit(X_train_indices, Y_train_oh, epochs=50, batch_size=32, verbose=1)
    
    # Extract the final output (e.g., accuracy)
    final_accuracy = history.history['accuracy'][-1]
    
    return final_accuracy

# Example usage
if __name__ == "__main__":
    # Example data (replace with actual data)
    vocab_size = 10000
    embedding_dim = 128
    max_sequence_length = 100
    num_classes = 10
    X_train_indices = [[...]...]  # Example input data
    Y_train_oh = [[...]...]  # Example one-hot encoded labels
    
    output = method()
    print(f"Final Accuracy: {output}")