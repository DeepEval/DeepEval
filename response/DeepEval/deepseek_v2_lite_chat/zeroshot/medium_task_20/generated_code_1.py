import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from keras.optimizers import Adam

# Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# Number of classes
num_classes = 10

# Define the input shape
input_shape = (32, 32, 3)

# Functional model creation
def dl_model():
    # Path 1: 1x1 convolution
    path1 = Conv2D(32, (1, 1), activation='relu', padding='same')(input_shape)
    
    # Path 2: 3x3 convolution stack
    path2 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_shape)
    path2 = Conv2D(64, (3, 3), activation='relu', padding='same')(path2)
    
    # Path 3: Single 3x3 convolution
    path3 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_shape)
    
    # Path 4: Max pooling followed by 1x1 convolution
    path4 = MaxPooling2D(pool_size=(2, 2))(input_shape)
    path4 = Conv2D(64, (1, 1), activation='relu', padding='same')(path4)
    
    # Concatenate the outputs from the paths
    concat = Concatenate()([path1, path2, path3, path4])
    
    # Flatten and pass through a dense layer
    dense = Flatten()(concat)
    dense = Dense(128, activation='relu')(dense)
    
    # Output layer
    output = Dense(num_classes, activation='softmax')(dense)
    
    # Create the model
    model = Model(inputs=input_shape, outputs=output)
    
    # Compile the model
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Build and return the model
model = dl_model()
model.summary()