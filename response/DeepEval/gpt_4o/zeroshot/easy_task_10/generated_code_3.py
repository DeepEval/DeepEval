from keras.layers import Input, Conv2D, DepthwiseConv2D, Flatten, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.utils import to_categorical
import keras

def dl_model():
    input_shape = (28, 28, 1)  # MNIST images are 28x28 pixels with a single color channel

    inputs = Input(shape=input_shape)

    # 1x1 Convolutional Layer with stride 2, to increase dimensionality
    x = Conv2D(filters=32, kernel_size=(1, 1), strides=(2, 2), activation='relu', padding='same')(inputs)

    # 3x3 Depthwise Separable Convolutional Layer
    x = DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(x)

    # 1x1 Convolutional Layer with stride 2, to reduce dimensionality
    x = Conv2D(filters=64, kernel_size=(1, 1), strides=(2, 2), activation='relu', padding='same')(x)

    # Flatten and Dense for classification
    x = Flatten()(x)
    outputs = Dense(10, activation='softmax')(x)  # 10 classes for MNIST

    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage:
if __name__ == "__main__":
    # Load and preprocess the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train[..., None]  # Add channel dimension
    x_test = x_test[..., None]  # Add channel dimension

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Create and compile the model
    model = dl_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Print the model summary
    model.summary()

    # Train the model
    model.fit(x_train, y_train, batch_size=32, epochs=10, validation_split=0.2)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("Test accuracy:", test_acc)