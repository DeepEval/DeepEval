from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from keras.datasets import cifar10
from keras.utils import to_categorical
import tensorflow as tf

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Function to create the model
def dl_model():
    # Branch path
    branch_input = Input(shape=(32, 32, 3))
    branch = Conv2D(32, (5, 5), activation='relu')(branch_input)
    branch = MaxPooling2D()(branch)

    # Main path
    main_input = Input(shape=(32, 32, 3))
    main = Conv2D(32, (3, 3), activation='relu')(main_input)
    main = MaxPooling2D()(main)

    # Concatenate the branch and main features
    concat = concatenate([branch, main])

    # Flatten and map to 10 classes
    output = Flatten()(concat)
    output = Dense(1024, activation='relu')(output)
    output = Dense(10, activation='softmax')(output)  # Assuming you want softmax activation for multi-class classification

    # Define the model
    model = Model(inputs=[branch_input, main_input], outputs=output)
    return model

# Create the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()