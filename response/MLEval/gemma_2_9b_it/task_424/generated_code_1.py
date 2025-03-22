import tensorflow as tf

def method():
    # Assuming you have a TensorFlow model named 'model'
    model = tf.keras.models.Sequential([
        # ... your model architecture ...
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model from scratch
    # ... your training code ...

    return model

# Call the method and store the output
output = method()

# Now you can use 'output' which is your newly trained model