from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def method():
    # Create a Sequential model
    model = Sequential()

    # Add a Dense layer with 64 units and ReLU activation
    model.add(Dense(64, activation='relu'))

    # Add an output layer with 1 unit and sigmoid activation
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Generate some sample data
    x_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_train = [0, 1, 1, 0]

    # Train the model
    model.fit(x_train, y_train, epochs=5)

    # Predict the output for a new sample
    output = model.predict([[0, 0]])

    return output

# Call the method and print the output
output = method()
print(output)