import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def method():
    # Generate dummy data
    x_train = np.random.random((1000, 20))
    y_train = np.random.randint(2, size=(1000, 1))

    x_val = np.random.random((200, 20))
    y_val = np.random.randint(2, size=(200, 1))

    # Create a simple Sequential model
    model = Sequential()
    model.add(Dense(64, input_dim=20, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer=Adam(), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])

    # Fit the model
    history = model.fit(x_train, y_train, 
                        epochs=10, 
                        batch_size=32, 
                        validation_data=(x_val, y_val))

    # Return the training history
    return history

# Call the method for validation
output = method()
print(output.history)