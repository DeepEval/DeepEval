import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

def method():
    # Load dataset (using Iris dataset for demonstration)
    data = load_iris()
    X = data.data
    y = to_categorical(data.target)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the model using Sequential
    model = Sequential()

    # Input layer
    model.add(Dense(64, input_shape=(X_train.shape[1],), activation='relu'))
    model.add(Dropout(0.5))  # Dropout layer to reduce overfitting

    # Hidden layer
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))  # Another dropout layer

    # Output layer
    model.add(Dense(y_train.shape[1], activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1, validation_split=0.2)

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    output = {'loss': loss, 'accuracy': accuracy}
    return output

# Call the method for validation
output = method()
print(output)