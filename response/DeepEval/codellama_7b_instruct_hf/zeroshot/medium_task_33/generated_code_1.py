from keras.layers import Input, Lambda, Dense, Dropout, Flatten, Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.applications.vgg16 import VGG16

# Define the input shape
input_shape = (32, 32, 3)

# Define the model
model = Sequential()
model.add(Lambda(lambda x: tf.split(x, 3, axis=3), output_shape=(3, 32, 32)))
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
model.add(SeparableConv2D(16, (1, 1), activation='relu'))
model.add(SeparableConv2D(16, (3, 3), activation='relu'))
model.add(SeparableConv2D(16, (5, 5), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(SeparableConv2D(32, (1, 1), activation='relu'))
model.add(SeparableConv2D(32, (3, 3), activation='relu'))
model.add(SeparableConv2D(32, (5, 5), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(SeparableConv2D(64, (1, 1), activation='relu'))
model.add(SeparableConv2D(64, (3, 3), activation='relu'))
model.add(SeparableConv2D(64, (5, 5), activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Return the constructed model
return model