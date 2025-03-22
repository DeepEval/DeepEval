import tensorflow as tf
from tensorflow import keras

def method():
    model = keras.Sequential([
        # Add your model architecture here
    ])

    # Compile the model
    model.compile(loss='categorical_crossentropy',  # Use categorical crossentropy for multi-class classification
                  optimizer='adam',
                  metrics=['accuracy'])

    # Fit the model
    history = model.fit(X_train_indices, Y_train_oh,
                        epochs=50,
                        batch_size=32)

    # You can access training history (e.g., loss, accuracy) using 'history'
    # For example:
    # print(history.history['accuracy'])

    return history

# Call the method to fit the model
output = method()