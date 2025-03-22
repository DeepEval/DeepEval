import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

# Example dataset
X = np.random.rand(1000, 20)  # 1000 samples, 20 features
y = np.random.randint(0, 2, size=(1000,))  # Binary target variable

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def method():
    # Define the model
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Assuming binary classification
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Fit the model using training data and validate using X_test, y_test
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # You can also return the model or its evaluation, depending on your needs
    output = model.evaluate(X_test, y_test)
    return output

# Call the method for validation
output = method()
print("Validation Output:", output)