import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Load and preprocess MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train[..., tf.newaxis]  # Add channel dimension
    x_test = x_test[..., tf.newaxis]
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Define the model architecture
    input_layer = Input(shape=(28, 28, 1))

    # 1x1 convolution to reduce dimensionality
    x = Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(input_layer)

    # 3x3 convolution to extract features
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)

    # 1x1 convolution to restore dimensionality
    x = Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(x)

    # Flatten the output
    x = Flatten()(x)

    # Fully connected layer with 10 neurons for classification
    output_layer = Dense(units=10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Example usage
model = dl_model()
model.summary()