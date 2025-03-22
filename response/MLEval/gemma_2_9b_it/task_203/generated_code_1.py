import tensorflow as tf

def method():
    # Define the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, dilation_rate=2, padding='causal', activation='relu', input_shape=(100, 1)), # Adjust input_shape to your data
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, dilation_rate=4, padding='causal', activation='relu'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(10, activation='softmax') # Adjust output units based on your task
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Validate the method
model = method()
print(model.summary())