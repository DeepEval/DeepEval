import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, AveragePooling2D, Flatten, Dense, Reshape, Multiply, concatenate
from keras.layers import GlobalAveragePooling2D
from keras.layers import Concatenate
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import InputLayer
from keras.optimizers import Adam

# Number of classes in CIFAR-10
NUM_CLASSES = 10

# Reshape and normalize input images
input_shape = (32, 32, 3)
input_image = Input(shape=input_shape)

# First convolutional layer
x = Conv2D(32, (3, 3), padding='same')(input_image)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# Second convolutional layer
x = Conv2D(64, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# Global average pooling
x = GlobalAveragePooling2D()(x)

# Fully connected layer 1
x = Dense(512, activation='relu')(x)

# Fully connected layer 2
x = Dense(NUM_CLASSES, activation='softmax')(x)

# Model
model = Model(inputs=input_image, outputs=x)

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

return model