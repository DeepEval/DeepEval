import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense
from keras.models import Model

# Define the input shape
input_shape = (32, 32, 3)

# Define the first convolutional layer
conv1 = Conv2D(32, (3, 3), activation='relu')

# Define the second convolutional layer
conv2 = Conv2D(64, (3, 3), activation='relu')

# Define the third convolutional layer
conv3 = Conv2D(128, (3, 3), activation='relu')

# Define the first max pooling layer
pool1 = MaxPooling2D((2, 2))

# Define the second max pooling layer
pool2 = MaxPooling2D((2, 2))

# Define the third max pooling layer
pool3 = MaxPooling2D((2, 2))

# Define the first concatenation layer
concatenation1 = Add()([conv1, pool1])

# Define the second concatenation layer
concatenation2 = Add()([conv2, pool2])

# Define the third concatenation layer
concatenation3 = Add()([conv3, pool3])

# Define the flatten layer
flatten = Flatten()(concatenation3)

# Define the first fully connected layer
fc1 = Dense(64, activation='relu')

# Define the second fully connected layer
fc2 = Dense(32, activation='relu')

# Define the output layer
output = Dense(10, activation='softmax')

# Define the model
model = Model(inputs=input_shape, outputs=output)
model.add(conv1)
model.add(pool1)
model.add(conv2)
model.add(pool2)
model.add(conv3)
model.add(pool3)
model.add(concatenation1)
model.add(concatenation2)
model.add(concatenation3)
model.add(flatten)
model.add(fc1)
model.add(fc2)
model.add(output)


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))