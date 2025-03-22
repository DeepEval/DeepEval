import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(28, 28, 1))

    main_path = keras.Sequential([
        layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
        layers.Dropout(rate=0.25),
        layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
        layers.Dropout(rate=0.25),
        layers.Conv2D(filters=10, kernel_size=(3,3), activation='relu'),
    ])(inputs)

    branch_path = keras.Sequential([
        layers.Conv2D(filters=10, kernel_size=(1,1), activation='relu'),
    ])(inputs)

    combined = layers.Add()([main_path, branch_path])

    outputs = layers.Flatten()(combined)
    outputs = layers.Dense(units=10, activation='softmax')(outputs)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', accuracy)