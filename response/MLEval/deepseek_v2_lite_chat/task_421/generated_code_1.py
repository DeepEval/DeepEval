import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def method():
    # Load your data
    # Assuming X_train_indices is your feature matrix and Y_train_oh is your target array
    # Also assuming Label Encoder for categorical variables if needed
    
    # Convert one-hot encoding back to original format if needed
    # Y_train_original = Y_train_oh.argmax(axis=1)

    # Normalize or standardize your data if needed
    
    # Split the data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_indices, Y_train_oh, test_size=0.2, random_state=42)

    # Define the model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(Y_train.shape[1], activation='softmax')  # Assuming Y_train is a binary classification problem
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_data=(X_val, Y_val))

    # Evaluate the model on the validation set
    val_loss, val_accuracy = model.evaluate(X_val, Y_val)

    return val_loss, val_accuracy

# Call the method for validation
val_loss, val_accuracy = method()
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")