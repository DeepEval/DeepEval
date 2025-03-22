import tensorflow as tf

# Create a simple sequential model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(units=10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=1)

# Restart the model from the beginning
model.reset_states()

# Compile the model again
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model again
model.fit(x_train, y_train, epochs=1)

# Get the final output
output = model.predict(x_test)

# Call the method for validation
print(method())