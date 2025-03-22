import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from keras.optimizers import Adam

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Number of classes
num_classes = 10

# Function to create the model
def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # Convolutional layer 1
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    
    # Convolutional layer 2
    x = Conv2D(64, (3, 3), activation='relu')(x)
    
    # MaxPooling layer
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Flatten layer
    x = Flatten()(x)
    
    # Fully connected layer 1
    x = Dense(512, activation='relu')(x)
    
    # Fully connected layer 2
    x = Dense(128, activation='relu')(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Build the model
model = dl_model()

# Display model summary
model.summary()