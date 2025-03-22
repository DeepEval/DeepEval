from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def method():
    # Load MNIST dataset as an example (replace with your dataset)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # # Preprocess the data
    # x_train = x_train.reshape((x_train.shape[0], -1)).astype('float32') / 255
    # x_test = x_test.reshape((x_test.shape[0], -1)).astype('float32') / 255
    # y_train = to_categorical(y_train, 10)
    # y_test = to_categorical(y_test, 10)
    
    # Create a Sequential model
    model = Sequential()
    
    # Add layers to the model
    model.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],)))
    model.add(Dropout(0.2))  # Experiment with different dropout rates
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))  # Output layer for 10 classes
    
    # Compile the model
    model.compile(optimizer=Adam(),
                  loss=SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    
    # Train the model
#     model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
    
#     # Evaluate the model on test data
#     loss, accuracy = model.evaluate(x_test, y_test)
    
#     # Output results
#     output = {"loss": loss, "accuracy": accuracy}
#     return output

# # Call the method for validation
# output = method()
# print(output)
method()