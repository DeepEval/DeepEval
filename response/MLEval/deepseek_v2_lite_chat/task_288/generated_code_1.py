import numpy as np
from keras.models import Sequential
from keras.layers import Dense

def method(input_features, hidden_nodes, output_nodes):
    # Initialize the model
    model = Sequential()
    
    # Add the first layer with the number of input features and hidden_nodes
    model.add(Dense(hidden_nodes, input_dim=input_features, activation='relu'))
    
    # Add the second layer with the number of output nodes (assuming a classification problem)
    model.add(Dense(output_nodes, activation='softmax'))
    
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

# Example usage
# For demonstration, let's assume we have 4 input features and the output is a binary classification (2 classes)
input_features = 4
hidden_nodes = 5
output_nodes = 2

model = method(input_features, hidden_nodes, output_nodes)
# model.fit(X_train, y_train, epochs=10, batch_size=32)  # Assuming X_train and y_train are your data and labels

# validation
# model.evaluate(X_test, y_test)