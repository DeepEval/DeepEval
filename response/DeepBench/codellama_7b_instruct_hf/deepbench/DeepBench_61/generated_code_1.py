from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def create_model(input_shape, num_classes):
    # create model
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=input_shape))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

    return model

# example usage
if __name__ == "__main__":
    # create some sample input data
    X = np.random.rand(100, 28, 28)
    y = np.random.randint(0, 10, size=(100,))

    # create the model
    model = create_model((28, 28), 10)

    # train the model
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

    # evaluate the model
    loss, accuracy = model.evaluate(X, y)
    print(f'Loss: {loss:.3f}, Accuracy: {accuracy:.3f}')