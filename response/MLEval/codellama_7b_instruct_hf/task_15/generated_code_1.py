import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define the training data
X_train = ... # training data
y_train = ... # training labels

# Define the validation data
X_val = ... # validation data
y_val = ... # validation labels

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model on the validation data
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f'Validation loss: {val_loss:.3f}, accuracy: {val_accuracy:.3f}')

# Experiment with different batch sizes
batch_sizes = [32, 64, 128, 256]
for batch_size in batch_sizes:
    history = model.fit(X_train, y_train, epochs=10, batch_size=batch_size, validation_data=(X_val, y_val))
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print(f'Batch size: {batch_size}, validation loss: {val_loss:.3f}, accuracy: {val_accuracy:.3f}')

# Experiment with different number of epochs
epochs = [5, 10, 15, 20]
for epoch in epochs:
    history = model.fit(X_train, y_train, epochs=epoch, batch_size=32, validation_data=(X_val, y_val))
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print(f'Epochs: {epoch}, validation loss: {val_loss:.3f}, accuracy: {val_accuracy:.3f}')

# Return the final output
# return model