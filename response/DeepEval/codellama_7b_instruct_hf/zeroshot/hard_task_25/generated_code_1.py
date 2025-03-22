import keras
from keras.layers import Input, Conv2D, MaxPool2D, AveragePooling2D, Concatenate, UpSampling2D, Conv2DTranspose, Flatten, Dense

# Define the input shape
input_shape = (32, 32, 3)

# Define the main path
main_path = Sequential()
main_path.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
main_path.add(MaxPool2D((2, 2), strides=(2, 2)))
main_path.add(Conv2D(64, (3, 3), activation='relu'))
main_path.add(MaxPool2D((2, 2), strides=(2, 2)))
main_path.add(Conv2D(64, (3, 3), activation='relu'))

# Define the branch path
branch_path = Sequential()
branch_path.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
branch_path.add(MaxPool2D((2, 2), strides=(2, 2)))
branch_path.add(Conv2D(64, (3, 3), activation='relu'))
branch_path.add(AveragePooling2D((2, 2), strides=(2, 2)))
branch_path.add(Conv2D(64, (3, 3), activation='relu'))
branch_path.add(UpSampling2D((2, 2)))
branch_path.add(Conv2D(32, (3, 3), activation='relu'))

# Define the fusion layer
fusion_layer = Concatenate(axis=3)

# Define the model
model = Sequential()
model.add(main_path)
model.add(branch_path)
model.add(fusion_layer)
model.add(Conv2DTranspose(32, (3, 3), activation='relu'))
model.add(Conv2DTranspose(32, (3, 3), activation='relu'))
model.add(Conv2DTranspose(32, (3, 3), activation='relu'))
model.add(Conv2D(10, (1, 1), activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])