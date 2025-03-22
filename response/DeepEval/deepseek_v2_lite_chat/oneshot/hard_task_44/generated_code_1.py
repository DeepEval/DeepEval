import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Activation
from keras.models import Model
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.datasets import cifar10
from keras.utils import to_categorical

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Model parameters
input_shape = (32, 32, 3)
num_classes = 10

# First block for splitting and feature extraction
def first_block(input_tensor):
    split = Lambda(lambda x: K.split(x, 3, axis=-1))(input_tensor)
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(split[0])
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(split[1])
    conv3 = Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same')(split[2])
    pool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv3)
    return keras.layers.concatenate([conv1, conv2, pool1])

# Second block for branch processing
def second_block(input_tensor):
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv3)
    conv4 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')(pool2)
    return keras.layers.concatenate([conv4])

# Flattening and fully connected layers
def fully_connected(input_tensor):
    flat = Flatten()(input_tensor)
    dense1 = Dense(units=256, activation='relu')(flat)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output = Dense(units=num_classes, activation='softmax')(dense2)
    return output

# Model building
inputs = Input(shape=input_shape)
block1_output = first_block(inputs)
block2_output = second_block(block1_output)
model = fully_connected(block2_output)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Return the model
return model

# Instantiate the model
model = dl_model()

# Train the model
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=64)