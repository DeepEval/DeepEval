import keras
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Dense, Flatten, Add, Conv2D

# Define the input shape
input_shape = (32, 32, 3)

# Define the main path
main_path = Sequential()
main_path.add(GlobalAveragePooling2D(input_shape=input_shape))
main_path.add(Dense(128, activation='relu'))
main_path.add(Dense(10, activation='softmax'))

# Define the branch path
branch_path = Sequential()
branch_path.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
branch_path.add(Conv2D(16, (3, 3), activation='relu'))
branch_path.add(Conv2D(8, (3, 3), activation='relu'))
branch_path.add(Flatten())
branch_path.add(Dense(10, activation='softmax'))

# Define the model
model = Model(inputs=Input(shape=input_shape), outputs=Add()([main_path.output, branch_path.output]))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the summary of the model
model.summary()