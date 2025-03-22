from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def method():
    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(timesteps, features)))  
    model.add(LSTM(units=50))
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

# Example usage:
# Assuming 'timesteps' and 'features' are defined with your data's dimensions
model = method()
# You can then train the model on your data
model.fit(X_train, y_train, epochs=10, batch_size=32)  

# And evaluate it
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')