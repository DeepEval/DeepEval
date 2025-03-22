from keras.models import Sequential
from keras.layers import LSTM, Dense

def method():
    # Define the model
    model = Sequential()
    model.add(LSTM(50, input_shape=(None, 1)))
    model.add(Dense(1))

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Generate some sample data
    X = np.random.rand(100, 1)
    y = np.random.rand(100)

    # Train the model
    model.fit(X, y, epochs=10)

    # Use the model to make predictions
    output = model.predict(X)

    return output

# Call the method for validation
method()