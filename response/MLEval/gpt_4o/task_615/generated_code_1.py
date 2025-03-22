import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def method():
    # Generate some synthetic data for demonstration
    np.random.seed(0)
    X = np.random.rand(1000, 20)
    y = np.random.randint(0, 2, size=(1000, 1))

    # Splitting the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create a simple neural network with dropout to prevent overfitting
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.5),  # Apply dropout with a rate of 0.5
        Dense(32, activation='relu'),
        Dropout(0.5),  # Another dropout layer
        Dense(1, activation='sigmoid')  # Output layer
    ])

    # Compile the model
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=0)

    # Evaluate the model on test data
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    output = {
        'loss': loss,
        'accuracy': accuracy
    }
    
    return output

# Call the method for validation
result = method()
print("Test Loss:", result['loss'])
print("Test Accuracy:", result['accuracy'])