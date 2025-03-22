from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, concatenate
from keras.models import Model

# Define the input shape
input_shape = (32, 32, 3)

# Define the four parallel branches
path1 = Conv2D(64, (3, 3), activation='relu')(input_shape)
path2 = AveragePooling2D((2, 2))(path1)
path2 = Conv2D(64, (3, 3), activation='relu')(path2)
path3 = Conv2D(64, (3, 3), activation='relu')(path1)
path3 = Conv2D(64, (3, 3), activation='relu')(path3)
path4 = Conv2D(64, (3, 3), activation='relu')(path1)
path4 = Conv2D(64, (3, 3), activation='relu')(path4)

# Define the multi-scale feature fusion layer
msf = concatenate([path1, path2, path3, path4])

# Define the fully connected layer
fc = Dense(10, activation='softmax')(msf)

# Define the model
model = Model(inputs=input_shape, outputs=fc)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)