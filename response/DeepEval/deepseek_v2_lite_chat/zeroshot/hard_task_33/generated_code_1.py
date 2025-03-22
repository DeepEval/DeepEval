import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Convolution2D, DepthwiseConv2D, SeparableConv2D, Flatten, Dense, concatenate
from tensorflow.keras.utils import to_categorical

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Model parameters
input_shape = (28, 28, 1)
num_classes = 10

# Function to define the model architecture
def dl_model():
    # Input layer
    inputs = Input(shape=input_shape)

    # Define the three branches
    def branch(filters, kernel_size, strides):
        # 1x1 convolution to elevate dimension
        x = Convolution2D(filters, (1, 1), strides=(strides, 1), padding='same')(inputs)
        # 3x3 depthwise separable convolution
        x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same')(x)
        # 1x1 convolution to reduce dimension
        x = SeparableConv2D(filters, kernel_size, strides=(1, 1), padding='same')(x)
        return x

    # Define the model
    branch1 = branch(filters=64, kernel_size=(3, 3), strides=(1, 1))
    branch2 = branch(filters=64, kernel_size=(3, 3), strides=(1, 1))
    branch3 = branch(filters=64, kernel_size=(3, 3), strides=(1, 1))

    # Concatenate the outputs from the branches
    x = concatenate([branch1, branch2, branch3])

    # Flatten the output and apply a fully connected layer
    x = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Build the model
model = dl_model()

# Display the model summary
model.summary()