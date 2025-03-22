import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define the input shape
input_shape = (32, 32, 3)

# Lambda layer to split the input into three groups
x = Input(shape=input_shape)
split_x = Lambda(lambda x: tf.split(x, 3, axis=-1))(x)

# Define 1x1, 3x3, and 5x5 convolutional layers with different kernels
conv1 = Conv2D(32, (1, 1), padding='same', activation='relu')(split_x[0])
conv2 = Conv2D(32, (3, 3), padding='same', activation='relu')(split_x[1])
conv3 = Conv2D(32, (5, 5), padding='same', activation='relu')(split_x[2])

# Concatenate the outputs from the three groups
concat = Concatenate()([conv1, conv2, conv3])

# Flatten the concatenated feature map
flatten = Flatten()(concat)

# Fully connected layers for classification
fc1 = Dense(512, activation='relu')(flatten)
output = Dense(10, activation='softmax')(fc1)

# Create the model
model = Model(inputs=x, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Return the compiled model
return model

# Instantiate and return the model
model = dl_model()
print(model.summary())