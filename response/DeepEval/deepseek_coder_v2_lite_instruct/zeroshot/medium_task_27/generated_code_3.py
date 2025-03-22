import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Add, GlobalAveragePooling2D, Dense, Multiply
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    # Preprocess the data
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    # Define the input layer
    inputs = Input(shape=(32, 32, 3))
    
    # First convolutional branch with 3x3 kernel
    branch_3x3 = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    branch_3x3 = Conv2D(32, (3, 3), padding='same', activation='relu')(branch_3x3)
    
    # Second convolutional branch with 5x5 kernel
    branch_5x5 = Conv2D(32, (5, 5), padding='same', activation='relu')(inputs)
    branch_5x5 = Conv2D(32, (5, 5), padding='same', activation='relu')(branch_5x5)
    
    # Add the outputs of the two branches
    added = Add()([branch_3x3, branch_5x5])
    
    # Global average pooling
    gap = GlobalAveragePooling2D()(added)
    
    # Fully connected layers
    fc1 = Dense(128, activation='relu')(gap)
    fc2 = Dense(10, activation='softmax')(fc1)
    
    # Create the model
    model = Model(inputs=inputs, outputs=fc2)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Example usage
model = dl_model()
model.summary()