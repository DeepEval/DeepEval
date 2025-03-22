import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical

def method(X_train_indices, Y_train_oh, epochs=50, batch_size=32):
    # Load the data
    X_train = X_train_indices
    Y_train = Y_train_oh

    # Encode the labels
    encoder = LabelEncoder()
    Y_train_encoded = encoder.fit_transform(Y_train)

    # Convert the labels to categorical
    Y_train_categorical = to_categorical(Y_train_encoded)

    # Initialize the model
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(Y_train_categorical.shape[1], activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, Y_train_categorical, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    # Print the training and validation accuracy and loss
    print('Training accuracy:', history.history['accuracy'][-1])
    print('Training loss:', history.history['loss'][-1])
    print('Validation accuracy:', history.history['val_accuracy'][-1])
    print('Validation loss:', history.history['val_loss'][-1])

    # Return the trained model
    return model