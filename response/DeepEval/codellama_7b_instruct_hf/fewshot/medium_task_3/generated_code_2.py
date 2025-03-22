import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, InputLayer, Add, Flatten, Dense

# Create the model
model = Sequential()

# First block
model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))

# Second block
model.add(Conv2D(filters=64, kernel_size=(5,5), strides=(1,1)))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))

# Add the input to the outputs of both blocks
model.add(Add())

# Flatten the output
model.add(Flatten())

# Output layer
model.add(Dense(units=10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])