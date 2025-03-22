from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

def dl_model():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Preprocess the data
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)) / 255.0
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)) / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # Build the model
    model = models.Sequential([
        layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), input_shape=(28, 28, 1)),
        layers.Conv2D(64, (1, 1), padding='same', activation='relu'),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Train and evaluate the model
model = dl_model()
model.fit(x_train, y_train, epochs=10)
model.evaluate(x_test, y_test)