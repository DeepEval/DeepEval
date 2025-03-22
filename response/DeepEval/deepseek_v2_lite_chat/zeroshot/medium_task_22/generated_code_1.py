import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0

# Get the dimensions of a single image
img_width, img_height, img_channels = 32, 32, 3

# Define the input shape
input_shape = (img_width, img_height, img_channels)

# Function to create the multi-branch convolutional architecture
def dl_model():
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Branch 1: 3x3 convolutions
    branch1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    branch1 = Conv2D(32, (3, 3), activation='relu', padding='same')(branch1)
    branch1 = MaxPooling2D(pool_size=(2, 2))(branch1)
    
    # Branch 2: 1x1 conv -> 2x2 conv -> max pooling
    branch2 = Conv2D(32, (1, 1), activation='relu')(inputs)
    branch2 = Conv2D(32, (3, 3), activation='relu', padding='same')(branch2)
    branch2 = Conv2D(32, (3, 3), activation='relu')(branch2)
    branch2 = MaxPooling2D(pool_size=(2, 2))(branch2)
    
    # Branch 3: max pooling
    branch3 = MaxPooling2D(pool_size=(2, 2))(inputs)
    
    # Concatenate features from all branches
    concat = concatenate([branch1, branch2, branch3])
    
    # Flatten and pass through two fully connected layers
    flat = Flatten()(concat)
    dense1 = Dense(256, activation='relu')(flat)
    dense2 = Dense(10, activation='softmax')(dense1)
    
    # Define the model
    model = Model(inputs=inputs, outputs=dense2)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Build the model
model = dl_model()

# Display the model summary
model.summary()