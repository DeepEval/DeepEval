from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.applications import MNIST

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = MNIST.load_data()

# Normalize the inputs
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Define the specialized block
def specialized_block(input_tensor):
    # Depthwise separable convolutional layer
    x = Conv2D(32, (3, 3), activation='relu')(input_tensor)
    x = Dropout(0.2)(x)
    # 1x1 convolutional layer
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = Dropout(0.2)(x)
    return x

# Define the model
model = Sequential()
model.add(specialized_block(X_train))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=128)