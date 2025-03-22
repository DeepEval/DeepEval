import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

def method():
    # Sample data
    texts = [
        "This is a simple LSTM example",
        "We are testing a basic LSTM model",
        "No pre-trained embeddings are used here",
        "LSTM is a type of recurrent neural network"
    ]
    labels = [0, 1, 0, 1]  # Sample labels for binary classification

    # Tokenize the text
    tokenizer = Tokenizer(num_words=1000)  # We consider only the top 1000 words
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Padding sequences to ensure they have the same length
    max_sequence_length = 10
    data = pad_sequences(sequences, maxlen=max_sequence_length)

    # Create the LSTM model
    model = Sequential()
    model.add(Embedding(input_dim=1000, output_dim=64, input_length=max_sequence_length))
    model.add(LSTM(32))  # Add an LSTM layer with 32 units
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Fit the model on the data
    model.fit(data, np.array(labels), epochs=5, batch_size=2)

    # For demonstration, let's return the model summary
    output = model.summary()
    return output

# Call the method for validation
method()