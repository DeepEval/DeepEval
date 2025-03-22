import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Concatenate
from keras.layers import GlobalAveragePooling2D, Multiply
from keras.optimizers import Adam

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the input shape
input_shape = (32, 32, 3)  # Average size of CIFAR-10 images

# Function to create the model
def dl_model():
    # Input layer
    input_layer = Input(shape=input_shape)
    
    # Branch 1: Convolutional Layer with 3x3 kernel
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    
    # Branch 2: Convolutional Layer with 5x5 kernel
    conv2 = Conv2D(64, (5, 5), activation='relu')(input_layer)
    
    # Add the two branches
    add_layer = Concatenate()([conv1, conv2])
    
    # Max pooling layer
    pool = MaxPooling2D()(add_layer)
    
    # Flatten before going to fully connected layers
    flat = Flatten()(pool)
    
    # Fully connected layer 1
    fc1 = Dense(512, activation='relu')(flat)
    
    # Fully connected layer 2
    fc2 = Dense(10, activation='softmax')(fc1)
    
    # Attention weights
    attention_1 = Dense(1, activation='sigmoid')(fc1)
    attention_2 = Dense(1, activation='sigmoid')(fc2)
    
    # Multiply attention weights with respective branches
    weighted_output = Multiply()([add_layer, attention_1 * conv1, attention_2 * conv2])
    
    # Model
    model = Model(inputs=input_layer, outputs=weighted_output)
    
    # Compile the model
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Build the model
model = dl_model()

# Print the model summary
model.summary()