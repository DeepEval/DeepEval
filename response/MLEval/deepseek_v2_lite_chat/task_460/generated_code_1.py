from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

def method():
    # Load your data (assuming it's a time series dataset)
    # Replace 'dataset.csv' with the path to your dataset
    data = np.loadtxt('dataset.csv', delimiter=',')
    # Split the dataset into features (X) and target (y)
    X = data[:, :-1]
    y = data[:, -1]

    # Normalize the data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y.reshape(-1, 1))

    # Split the data into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the LSTM model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)),
        Dense(1)
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    # Make predictions
    y_pred = model.predict(X_test)

    # Invert the normalization for the predicted values
    y_pred_denorm = scaler.inverse_transform(y_pred)
    y_test_denorm = scaler.inverse_transform(y_test)

    # Plot the results
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, len(y_test_denorm) + 1), y_test_denorm.flatten(), label='Actual')
    plt.plot(range(1, len(y_pred_denorm) + 1), y_pred_denorm.flatten(), label='Predicted')
    plt.legend()
    plt.title('LSTM Prediction vs Actual')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.show()

    # Return the final output
    return y_pred_denorm

# Call the method for validation
output = method()
print("Predicted values:", output)