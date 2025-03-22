import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

# Load and preprocess CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# Define the input shape
input_shape = (32, 32, 3)

# Function to create the deep learning model
def dl_model():
    # Input layer
    input_layer = Input(shape=input_shape)
    
    # Convolutional layers
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    
    # MaxPooling layer
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Flatten and fully connected layers
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=x)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Get the constructed model
model = dl_model()