import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.optimizers import Adam

def method():
    # Define the model
    model = Sequential()
    
    # Add an Embedding layer
    model.add(Embedding(input_dim=10000, output_dim=128, input_length=100))
    
    # Add an LSTM layer
    model.add(LSTM(units=64, return_sequences=True))
    
    # Add a Dense layer for output
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    # Example input and output (for demonstration purposes)
    # In a real scenario, you would replace this with actual data
    X_train = tf.random.normal([500, 100])
    y_train = tf.random.normal([500, 1])
    
    # Train the model (for demonstration purposes, we don't actually train the model here)
    # model.fit(X_train, y_train, epochs=5, batch_size=32)
    
    # Return some output (for demonstration purposes)
    output = model.summary()
    
    return output

# Call the method for validation
if __name__ == "__main__":
    print(method())