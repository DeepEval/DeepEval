# Import necessary libraries
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Function to construct the deep learning model
def dl_model():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape input data and normalize pixel values
    x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
    x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0

    # One-hot encode labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Define input shape for the model
    input_shape = (28, 28, 1)

    # Define the depthwise separable convolutional layer
    x = Input(shape=input_shape)
    x = DepthwiseConv2D(32, (3, 3), padding='same', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.2)(x)

    # Define the 1x1 convolutional layer for feature extraction
    x = Conv2D(32, (1, 1), padding='same', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.2)(x)

    # Flatten the output
    x = Flatten()(x)

    # Define the fully connected layer for classification
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.2)(x)
    outputs = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=x, outputs=outputs)

    # Compile the model with Adam optimizer and sparse categorical crossentropy loss
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Construct and return the deep learning model
model = dl_model()
print(model.summary())