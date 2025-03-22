import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from keras.layers import Concatenate, GlobalAveragePooling2D
from keras.optimizers import Adam

# Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# Define the input shape
input_shape = (32, 32, 3)

# Define the model
def dl_model():
    # Block 1: Global Average Pooling and Fully Connected Layers
    input_layer = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D()(x)
    x = GlobalAveragePooling2D()(x)
    fc1 = Dense(512, activation='relu')(x)
    fc2 = Dense(10, activation='softmax')(fc1)
    
    # Block 2: Convolutional Layers for Feature Extraction
    x = Conv2D(64, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D()(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D()(x)
    
    # Branch from Block 1 to Block 2
    branch = GlobalAveragePooling2D()(fc1)
    
    # Concatenate the main path and the branch
    output = Concatenate()([x, branch])
    
    # Fully Connected Layers for Classification
    fc3 = Dense(512, activation='relu')(output)
    output = Dense(10, activation='softmax')(fc3)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output)
    
    # Compile the model
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Build and return the model
model = dl_model()
model.summary()