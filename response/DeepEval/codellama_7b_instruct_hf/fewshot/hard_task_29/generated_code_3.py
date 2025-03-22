import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

# Define the input shape
input_shape = (28, 28, 1)

# Define the model
model = keras.Sequential()

# Block 1: Main Path
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Block 1: Branch Path
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Block 2: Main Path
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Block 2: Branch Path
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Combine the outputs of both blocks
model.add(Add())

# Flatten the output
model.add(Flatten())

# Add a fully connected layer
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])