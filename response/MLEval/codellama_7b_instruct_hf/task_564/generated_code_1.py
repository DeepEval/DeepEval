from keras.layers import LSTM, Dense
from keras.models import Model
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Generate some sample data
input_data = np.random.rand(100, 10)
target_data = np.random.randint(0, 10, 100)

# Create the LSTM layer
lstm_layer = LSTM(units=10, return_sequences=True)

# Create the model
model = Model(inputs=lstm_layer.input, outputs=lstm_layer.output)

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(input_data, target_data, epochs=10, batch_size=32)

# Use the model to make predictions on a new sequence of data
new_data = np.random.rand(10, 10)
output = model.predict(new_data)

print(output)