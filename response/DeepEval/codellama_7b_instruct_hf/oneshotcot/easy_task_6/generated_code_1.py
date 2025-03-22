import keras
import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the images
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Define the first path of the model
def first_path(input_tensor):
    x = keras.layers.Conv2D(32, (3, 3), activation='relu')(input_tensor)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    return x

# Define the branch path
def branch_path(input_tensor):
    x = keras.layers.Conv2D(16, (1, 1), activation='relu')(input_tensor)
    return x

# Combine the two paths through an addition operation
def combine_paths(first_path_output, branch_path_output):
    x = first_path_output + branch_path_output
    return x

# Flatten the output of the addition operation
def flatten_output(x):
    x = keras.layers.Flatten()(x)
    return x

# Add a fully connected layer to produce the final classification results
def fully_connected(x):
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dense(10, activation='softmax')(x)
    return x

# Define the model
model = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(28, 28, 1)),
    first_path,
    branch_path,
    combine_paths,
    flatten_output,
    fully_connected
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

# Use the model to predict the class of a new image
img_path = 'path/to/image.jpg'
img = keras.preprocessing.image.load_img(img_path, target_size=(28, 28))
img_array = keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
preds = model.predict(img_array)
print('Predicted class:', np.argmax(preds))