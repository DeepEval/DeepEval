import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def method():
    # Generate some dummy data
    X_train = np.random.rand(100, 10)  # 100 samples, 10 features
    y_train = np.random.rand(100, 1)   # 100 samples, 1 target variable

    # Create a Sequential model
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(10,)))  # Input layer
    model.add(Dense(64, activation='relu'))                     # Hidden layer
    model.add(Dense(1, activation='linear'))                     # Output layer

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=8, verbose=1)

    # You can return the model or its history if needed
    output = model
    return output

# Call the method for validation
trained_model = method()