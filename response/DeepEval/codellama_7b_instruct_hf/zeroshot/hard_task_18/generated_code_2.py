import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

# Define the input shape
input_shape = (32, 32, 3)

# Define the first block
block1 = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten()
])

# Define the second block
block2 = Sequential([
    GlobalAveragePooling2D(),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Flatten()
])

# Define the main path
main_path = Sequential([
    block1,
    block2
])

# Define the input and output layers
input_layer = Input(shape=input_shape)
output_layer = main_path(input_layer)

# Define the model
model = keras.Model(inputs=input_layer, outputs=output_layer)


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))