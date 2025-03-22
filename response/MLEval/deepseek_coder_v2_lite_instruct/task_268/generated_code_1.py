import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def method():
    # Define the model
    model = Sequential()
    
    # Add input layer and first hidden layer
    model.add(Dense(10, input_dim=4, activation='relu'))
    
    # Add additional hidden layer
    model.add(Dense(6, activation='relu'))
    
    # Add output layer
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Dummy data for demonstration (replace with actual data)
    X_train = tf.random.normal([100, 4])
    y_train = tf.random.uniform([100], 0, 2, dtype=tf.int32)
    
    # Train the model (dummy training for demonstration purposes)
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    
    # Dummy input for final output (replace with actual input data)
    X_test = tf.random.normal([1, 4])
    
    # Get the final output
    output = model.predict(X_test)
    
    return output

# Call the method for validation
output = method()
print(output)