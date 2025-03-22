import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from tensorflow.keras.regularizers import l2


# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Input shape for the model
input_shape = (32, 32, 3)  # Each image is 32x32 pixels


def branch1():
    return Conv2D(32, (1, 1), activation='relu', kernel_regularizer=l2(0.0005))(input_layer)

def branch2():
    return Conv2D(32, (1, 7), activation='relu', padding='valid', kernel_regularizer=l2(0.0005))(input_layer)

def branch3():
    return MaxPooling2D(pool_size=(3, 3))(input_layer)

def branch_concatenate(branch1, branch2, branch3):
    return concatenate([branch1, branch2, branch3])

def final_output(branch_concatenate):
    return Flatten()(branch_concatenate)

    # Add fully connected layers
    fc1 = Dense(1024, activation='relu', kernel_regularizer=l2(0.0005))(final_output)
    fc2 = Dense(512, activation='relu', kernel_regularizer=l2(0.0005))(fc1)
    output = Dense(10, activation='softmax')(fc2)

    # Return the complete model
    return Model(inputs=input_layer, outputs=output)

# Create the model
input_layer = Input(shape=input_shape)
model = final_output(branch_concatenate(branch1(), branch2(), branch3()))
model = Dense(1024, activation='relu')(model)
model = Dense(512, activation='relu')(model)
output = Dense(10, activation='softmax')(model)


model = tf.keras.models.Model(inputs=input_layer, outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

return model