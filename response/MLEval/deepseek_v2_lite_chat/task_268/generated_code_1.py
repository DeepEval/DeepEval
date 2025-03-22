import keras
from keras.models import Sequential
from keras.layers import Dense

def method():
    # Step 1: Define the input shape
    input_shape = (10,)  # Assuming the input has 10 features

    # Step 2: Create a sequential model
    model = Sequential()

    # Step 3: Add the first hidden layer
    model.add(Dense(4, input_shape=input_shape, activation='relu'))

    # Step 4: Add the additional hidden layer
    model.add(Dense(6, activation='relu'))

    # Optionally, add a final output layer if the problem requires one
    # model.add(Dense(1, activation='sigmoid'))  # For binary classification
    # model.add(Dense(10, activation='softmax'))  # For multi-class classification

    # Step 5: Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])  # For regression
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # For classification

    # Step 6: Return the final output
    return model

# Call the method for validation
model = method()