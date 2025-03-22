import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense
from keras.layers import BatchNormalization

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape the data for the convolutional layers
input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32
x_train = x_train.reshape(x_train.shape[0], *input_shape)
x_test = x_test.reshape(x_test.shape[0], *input_shape)

# Parameters
num_classes = 10
input_shape = (32, 32, 3)

# Model building function
def dl_model():
    # Input layer
    input_img = Input(shape=input_shape)

    # First convolutional layer
    x = Conv2D(32, (3, 3), activation='relu')(input_img)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    # Second convolutional layer
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    # Third convolutional layer
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)

    # Separate convolutional layer for input
    x = Conv2D(64, (3, 3), activation='relu')(input_img)
    x = BatchNormalization()(x)

    # Concatenate all outputs
    x = Concatenate()([x, x, x])

    # Flatten
    x = Flatten()(x)

    # Fully connected layer 1
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)

    # Fully connected layer 2
    x = Dense(num_classes, activation='softmax')(x)

    # Model
    model = Model(inputs=input_img, outputs=x)

    return model

# Build the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()