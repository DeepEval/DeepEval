from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.optimizers import Adam
import numpy as np

def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Normalize pixel values to be between 0 and 1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Convert class vectors to binary class matrices
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    # Input shape
    input_shape = (32, 32, 3)
    
    # Model input
    inputs = Input(shape=input_shape)
    
    # Define a basic block
    def basic_block(x, filters):
        y = Conv2D(filters, (3, 3), padding='same')(x)
        y = BatchNormalization()(y)
        y = ReLU()(y)
        y = Conv2D(filters, (3, 3), padding='same')(y)
        y = BatchNormalization()(y)
        return Add()([y, x])
    
    # First block
    x = basic_block(inputs, 16)
    
    # Second block
    x = basic_block(x, 32)
    
    # Branch for feature extraction
    branch_output = basic_block(inputs, 64)(inputs)
    
    # Feature fusion
    x = Add()([x, branch_output])
    
    # Reduce dimensionality
    x = Conv2D(64, (1, 1))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Classification head
    x = AveragePooling2D(pool_size=(8, 8))(x)
    x = Flatten()(x)
    outputs = Dense(10, activation='softmax')(x)
    
    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Create and return the model
model = dl_model()