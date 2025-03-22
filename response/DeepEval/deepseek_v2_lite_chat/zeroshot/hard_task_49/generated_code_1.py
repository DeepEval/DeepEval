import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Lambda, concatenate, Reshape

def dl_model():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values to [0, 1]

    # Define the first block
    block1 = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(1, 1), strides=(1, 1)),
        MaxPooling2D(pool_size=(2, 2), strides=(1, 1)),
        MaxPooling2D(pool_size=(4, 4), strides=(1, 1)),
        Flatten(),
        Lambda(lambda x: tf.concat([x, x, x], axis=-1) if x.shape[0] == 1 else x),  # Concatenate along the channel dimension
    ])

    # Define the transformation from the first block to the second block
    second_input_shape = (int(x_train.shape[1]/8), int(x_train.shape[2]/8), 4)  # 4-dimensional tensor
    block1_output_shape = (int(x_train.shape[1]/8), int(x_train.shape[2]/8), 4)  # Expected shape for second block input
    reshape_layer = tf.keras.layers.Reshape(target_shape=second_input_shape)(block1.output)

    # Define the second block
    block2 = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=reshape_layer.shape[1:]),
        Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=-1)),  # Split into four groups
        Conv2D(filters=48, kernel_size=(1, 1), activation='relu'),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        Conv2D(filters=96, kernel_size=(5, 5), activation='relu'),
        Conv2D(filters=128, kernel_size=(7, 7), activation='relu'),
        concatenate([g1, g2, g3, g4]),  # Concatenate the processed groups
        Flatten(),
        Dense(units=10)  # Output layer
    ])

    # Return the combined model
    model = tf.keras.models.Model(inputs=block1.input, outputs=block2.output)
    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)