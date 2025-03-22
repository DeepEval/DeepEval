# Import necessary packages
from tensorflow.keras.layers import Input, Concatenate, Dropout, Dense
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize input data
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Define the block
def block(inputs):
    x = DepthwiseConv2D((3, 3), activation='relu', padding='same')(inputs)
    x = Dropout(0.2)(x)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = Dropout(0.2)(x)
    return x

# Define the deep learning model
def dl_model():
    # Define the input shape
    inputs = Input(shape=(28, 28, 1))
    
    # Apply the block three times
    branch1 = block(inputs)
    branch2 = block(inputs)
    branch3 = block(inputs)
    
    # Concatenate the outputs from the three branches
    x = Concatenate()([branch1, branch2, branch3])
    
    # Apply two fully connected layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(10, activation='softmax')(x)
    
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Create and return the model
model = dl_model()