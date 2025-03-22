from keras.models import Model
from keras.layers import Input, Conv2D, Add, concatenate, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
from keras.applications.vgg16 import VGG16
from keras.datasets import cifar10
from keras.utils import to_categorical
import numpy as np

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Number of classes
num_classes = y_train.shape[1]

# VGG16 features for transfer learning
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # First branch
    branch1 = base_model(inputs)
    
    # Second branch with multiple 1x1, 1x3, and 3x1 convolutions
    branch2 = Conv2D(filters=64, kernel_size=(1, 3), activation='relu', padding='valid')(inputs)
    branch2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='valid')(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(3, 1), activation='relu', padding='valid')(branch2)
    
    # Concatenate the outputs of the two branches
    concat = concatenate([branch1, branch2])
    
    # Third convolution to match the input channel dimensions
    conv_out = Conv2D(filters=3, kernel_size=(1, 1), activation='sigmoid')(concat)
    
    # Direct connection from input to the branch
    input_branch = Conv2D(filters=3, kernel_size=(1, 1), activation='relu', name='direct_connection')(inputs)
    
    # Add the direct connection to the main branch
    model = Add()([input_branch, concat])
    
    # Flatten the output for fully connected layers
    flat = Flatten()(model)
    
    # Two fully connected layers for classification
    outputs = Dense(num_classes, activation='softmax')(flat)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False
    
    return model

# Build the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()