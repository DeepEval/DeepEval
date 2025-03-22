import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def method():
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=(sequence_length, input_dim)))
    model.add(LSTM(units=64))
    model.add(Dense(units=output_dim, activation='softmax')) 

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Example usage:
if __name__ == "__main__":
    sequence_length = 100  # Example sequence length
    input_dim = 50      # Example input dimension
    output_dim = 10     # Example output dimension

    model = method()
    print(model.summary())