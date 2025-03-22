import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Flatten

# Define the input shape
input_shape = (32, 32, 3)

# Define the model
model = keras.Sequential()
model.add(GlobalAveragePooling2D(input_shape=input_shape))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model
model.evaluate(X_test, y_test)