from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Concatenate
from keras.layers import BatchNormalization, LeakyReLU, Activation
from keras.layers import Flatten, Dense
from keras.datasets import cifar10
from keras.utils import to_categorical

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Get the dimensions of a single image
img_width, img_height, channels = 32, 32, 3

# Input tensor
inputs = Input(shape=(img_width, img_height, channels))

# Stage 1: Convolution and Max Pooling
x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Stage 2: Additional Convolution and Dropout
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

# Skip connections by concatenating features from corresponding layers
x1 = Conv2D(64, (1, 1), activation='relu')(x)
x = concatenate([x, x1])

# Additional convolutional layers and dropout
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Restore spatial information using UpSampling2D and concatenation
x2 = Conv2D(64, (1, 1), activation='relu')(x)
x = concatenate([x, x2])

# Additional upsampling layers with skip connections
x = Conv2D(64, (3, 3), activation='relu')(x)
x = UpSampling2D(size=2)(x)
x = concatenate([x, x2])

x = Conv2D(32, (3, 3), activation='relu')(x)
x = UpSampling2D(size=2)(x)

# Output layer
outputs = Conv2D(10, (1, 1), activation='softmax')(x)

# Define the model
model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

return model