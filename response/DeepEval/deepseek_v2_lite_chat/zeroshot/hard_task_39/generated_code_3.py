import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, concatenate, Reshape

def dl_model():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Reshape the data
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))

    # First block: max pooling layers and flattening
    model = tf.keras.Sequential([
        MaxPooling2D(pool_size=(1, 1), strides=(1, 1)),  # 1x1
        MaxPooling2D(pool_size=(2, 2), strides=(1, 1)),  # 2x2
        MaxPooling2D(pool_size=(4, 4), strides=(1, 1)),  # 4x4
        Flatten(),
    ])

    # Second block: multiple branches
    model.add(Conv2D(32, kernel_size=(1, 1), activation='relu'))  # 1x1 conv
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))   # 3x3 conv
    model.add(Conv2D(128, kernel_size=(5, 5), activation='relu'))  # 5x5 conv
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))      # 3x3 max pooling

    # Concatenate the outputs from all branches
    output = concatenate([model.output, model.output])

    # Reshape for final classification
    output = Reshape((-1,))(output)

    # Fully connected layer for classification
    model = Dense(10, activation='softmax')(output)

    # Return the model
    return Model(inputs=model.input, outputs=model)

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)