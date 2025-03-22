import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
from tensorflow.keras.utils import to_categorical

# Assuming X_train_indices and Y_train_oh are defined elsewhere
# X_train_indices = ...  # Your training input data (e.g., word indices)
# Y_train_oh = ...       # Your one-hot encoded training labels

def method():
    # Define the model
    model = Sequential()
    model.add(Embedding(input_dim=10000, output_dim=128))  # Example input_dim and output_dim
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(Y_train_oh.shape[1], activation='softmax'))  # Output layer for classification

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Fit the model
    model.fit(X_train_indices, Y_train_oh, epochs=50, batch_size=32)

    # Optionally return the model or its history for validation
    return model

# Call the method for validation
output = method()