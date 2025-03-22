import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Add, GlobalAveragePooling2D, Dense, Multiply
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    # Normalize the input data
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First convolutional branch: 3x3 kernel
    branch1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    branch1 = Conv2D(32, (3, 3), activation='relu', padding='same')(branch1)
    
    # Second convolutional branch: 5x5 kernel
    branch2 = Conv2D(32, (5, 5), activation='relu', padding='same')(input_layer)
    branch2 = Conv2D(32, (5, 5), activation='relu', padding='same')(branch2)
    
    # Combine the outputs of the two branches through addition
    combined = Add()([branch1, branch2])
    
    # Global average pooling
    pooled = GlobalAveragePooling2D()(combined)
    
    # Fully connected layers
    fc1 = Dense(128, activation='relu')(pooled)
    fc2 = Dense(10, activation='softmax')(fc1)
    
    # Compute attention weights
    attention_weights = Dense(1, activation='sigmoid')(fc1)
    
    # Produce the final weighted output
    weighted_output = Multiply()([attention_weights, fc2])
    
    # Define the model
    model = Model(inputs=input_layer, outputs=weighted_output)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Create and compile the model
model = dl_model()