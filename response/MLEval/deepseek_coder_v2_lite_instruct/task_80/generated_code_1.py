import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.models import Sequential

def method():
    model = Sequential([
        Conv2D(32, (3, 3), input_shape=(28, 28, 1)),
        BatchNormalization(),
        Activation('relu'),
        Dense(128, activation='relu'),
        Conv2D(64, (3, 3)),
        BatchNormalization(),
        Activation('relu'),
        Dense(64, activation='relu'),
        Conv2D(128, (3, 3)),
        BatchNormalization(),
        Activation('relu'),
        Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Assuming we have some input data for the model
    input_data = tf.random.normal((1, 28, 28, 1))

    # Get the output of the final layer
    output = model(input_data, training=True)

    return output

# Call the method for validation
output = method()
print(output)