# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert labels to categorical
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the deep learning model using Functional APIs of Keras
def dl_model():
    # Input layer
    inputs = keras.Input(shape=(28, 28))

    # 1x1 convolutional layer to increase dimensionality
    x = layers.Conv2D(32, (1, 1), strides=2, activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    # 3x3 depthwise separable convolutional layer for feature extraction
    x = layers.DepthwiseConv2D((3, 3), strides=2, activation='relu')(x)
    x = layers.SeparableConv2D(32, (1, 1))(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # 1x1 convolutional layer to reduce dimensionality
    x = layers.Conv2D(64, (1, 1), strides=2, activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Flattening layer
    x = layers.Flatten()(x)

    # Fully connected layer for classification
    outputs = layers.Dense(10, activation='softmax')(x)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

# Compile and train the model
model = dl_model()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {accuracy:.2f}")