import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

# Define the input shape
input_shape = (32, 32, 3)

# Define the model
model = keras.Sequential()

# Add the convolutional layers
model.add(Conv2D(32, (1, 1), activation='relu', input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(32, (5, 5), activation='relu'))

# Add the pooling layers
model.add(MaxPooling2D((2, 2)))
model.add(MaxPooling2D((3, 3)))
model.add(MaxPooling2D((5, 5)))

# Add the concatenation layer
model.add(Concatenate())

# Add the flattening layer
model.add(Flatten())

# Add the fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))

# Add the output layer
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the summary of the model
print(model.summary())