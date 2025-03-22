import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model

# Load and preprocess CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# Define the input shape and channels
input_shape = (32, 32, 3)
num_classes = 10

# Define the input layer
inputs = Input(shape=input_shape)

# Split the input into three channel groups
split1 = Lambda(lambda x: tf.split(x, 3, axis=-1))(inputs)

# Feature extraction for the first group using separable convolutional layers
conv1 = split1[0]
conv1 = Conv2D(32, (3, 3), activation='relu')(conv1)
conv1 = Conv2D(32, (3, 3), activation='relu')(conv1)
conv1 = MaxPooling2D(pool_size=(2, 2))(conv1)

split2 = split1[1:]
split2 = Lambda(lambda x: tf.split(x, 3, axis=-1))(split2)

# Feature extraction for the second group using separable convolutional layers
conv2 = split2[0]
conv2 = Conv2D(64, (3, 3), activation='relu')(conv2)
conv2 = Conv2D(64, (3, 3), activation='relu')(conv2)
conv2 = MaxPooling2D(pool_size=(2, 2))(conv2)

split3 = split2[1:]
split3 = Lambda(lambda x: tf.split(x, 3, axis=-1))(split3)

# Feature extraction for the third group using separable convolutional layers
conv3 = split3[0]
conv3 = Conv2D(64, (3, 3), activation='relu')(conv3)
conv3 = Conv2D(64, (3, 3), activation='relu')(conv3)
conv3 = MaxPooling2D(pool_size=(2, 2))(conv3)

# Concatenate the outputs from the three groups
x = tf.concat([conv1, conv2, conv3], axis=-1)

# Pass through three fully connected layers
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(num_classes, activation='softmax')(x)

# Define the model
model = Model(inputs=inputs, outputs=x)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

return model

# To use the model
# model = dl_model()
# model.summary()