from keras.layers import Dense, Dropout
from keras.models import Sequential

def method():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))

    # Add a layer after each dense layer
    for layer in model.layers:
        if isinstance(layer, Dense):
            model.pop(layer)
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.2))

    return model

# Call the method for validation
model = method()