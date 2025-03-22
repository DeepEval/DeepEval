import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Add, GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.optimizers import Adam

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Function to create the deep learning model
def dl_model():
    # Input layers
    input_layer = Input(shape=x_train[0].shape)
    
    # Branch 1 - Conv Layer with 3x3 kernel
    branch1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    branch1 = Conv2D(64, (3, 3), activation='relu', padding='same')(branch1)
    branch1 = GlobalAveragePooling2D()(branch1)
    
    # Branch 2 - Conv Layer with 5x5 kernel
    branch2 = Conv2D(32, (5, 5), activation='relu', padding='same')(input_layer)
    branch2 = Conv2D(64, (5, 5), activation='relu', padding='same')(branch2)
    branch2 = GlobalAveragePooling2D()(branch2)
    
    # Add the features from both branches
    add_layer = Add()([branch1, branch2])
    
    # Flatten and fully connected layers
    flat = Flatten()(add_layer)
    
    # Fully connected layer with softmax function to generate attention weights
    dense1 = Dense(128, activation='relu')(flat)
    dense1 = Dense(64, activation='relu')(dense1)
    
    # Softmax layer
    softmax = Dense(10, activation='softmax')(dense1)
    
    # Model
    model = Model(inputs=input_layer, outputs=softmax)
    
    # Compile the model
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Build and return the model
model = dl_model()
print(model.summary())