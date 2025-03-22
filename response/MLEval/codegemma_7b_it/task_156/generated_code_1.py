import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Flatten

def method():

    model = Sequential()
    model.add(LSTM(128, activation='relu', input_shape=(max_length,)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x_train_padded, y_train_encoded, epochs=10)
    output = model.evaluate(x_test_padded, y_test_encoded)

    return output

# Example usage:
output = method()
print(output)