import tensorflow as tf
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Dense, Reshape, Lambda, SeparableConv2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # First Block: Average Pooling with varying scales
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    flat1 = Flatten()(pool1)

    pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    flat2 = Flatten()(pool2)

    pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)
    flat3 = Flatten()(pool3)

    # Concatenate flattened results
    concatenated = Concatenate()([flat1, flat2, flat3])

    # Fully connected layer and reshape operation
    fc = Dense(units=256, activation='relu')(concatenated)
    reshaped = Reshape(target_shape=(4, 4, 16))(fc)  # Reshaping to a 4D tensor

    # Second Block: Splitting and depthwise separable convolutional layers
    split_layers = Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=-1))(reshaped)

    # Define separate convolution operations for each split
    conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_layers[0])
    conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_layers[1])
    conv3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_layers[2])
    conv4 = SeparableConv2D(filters=32, kernel_size=(7, 7), padding='same', activation='relu')(split_layers[3])

    # Concatenate results from convolutions
    concatenated_conv = Concatenate()([conv1, conv2, conv3, conv4])

    # Flatten and output layer
    flattened_output = Flatten()(concatenated_conv)
    output_layer = Dense(units=10, activation='softmax')(flattened_output)  # MNIST has 10 classes

    # Construct model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage:
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32') / 255
x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Train the model
model.fit(x_train, y_train, batch_size=128, epochs=10, validation_split=0.2)