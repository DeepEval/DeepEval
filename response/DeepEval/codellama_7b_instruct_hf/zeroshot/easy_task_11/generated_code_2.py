import keras
from keras.models import Model
from keras.layers import Input, Flatten, Dropout, Dense
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling2D

# Define the input shape
input_shape = (28, 28, 1)

# Define the model
model = Sequential()
model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), input_shape=input_shape))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

return model