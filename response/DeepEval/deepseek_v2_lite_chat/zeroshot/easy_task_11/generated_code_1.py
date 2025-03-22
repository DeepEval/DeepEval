import keras
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the data
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

# Normalize inputs from 0-255 to 0-1
x_train /= 255
x_test /= 255

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Function to create the deep learning model
def dl_model():
    # Input layer
    input_layer = keras.layers.Input(shape=(28, 28, 1))
    
    # Conv2D layer with 5x5 window and 3x3 stride
    conv1 = Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same')(input_layer)
    
    # MaxPooling2D layer
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Conv2D layer to enhance depth
    conv2 = Conv2D(64, kernel_size=(1, 1), activation='relu', padding='same')(pool1)
    
    # Flatten layer
    flat1 = Flatten()(conv2)
    
    # Fully connected layer
    dense1 = Dense(256, activation='relu')(flat1)
    
    # Dropout layer to mitigate overfitting
    drop1 = Dropout(0.5)(dense1)
    
    # Second fully connected layer
    dense2 = Dense(10, activation='softmax')(drop1)
    
    # Model building
    model = Model(inputs=input_layer, outputs=dense2)
    
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    
    return model

# Build the model
model = dl_model()

# Display model summary
model.summary()