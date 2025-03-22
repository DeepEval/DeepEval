import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Define the input shape based on CIFAR-10 images
    input_shape = (32, 32, 3)
    inputs = Input(shape=input_shape)
    
    # First branch with 3x3 convolutions
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x1)
    
    # Second branch with 1x1 convolution followed by two 3x3 convolutions
    x2 = Conv2D(32, (1, 1), activation='relu', padding='same')(inputs)
    x2 = Conv2D(32, (3, 3), activation='relu', padding='same')(x2)
    x2 = Conv2D(32, (3, 3), activation='relu', padding='same')(x2)
    
    # Third branch with max pooling
    x3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    
    # Concatenate the outputs from the three branches
    concatenated = Concatenate()([x1, x2, x3])
    
    # Flatten the concatenated feature maps
    x = Flatten()(concatenated)
    
    # Two fully connected layers for classification
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    
    # Output layer for classification (10 classes for CIFAR-10)
    outputs = Dense(10, activation='softmax')(x)
    
    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize to range [0, 1]
y_train, y_test = to_categorical(y_train), to_categorical(y_test)  # One-hot encode the labels

# Create the model
model = dl_model()

# Print the model summary
model.summary()

# Optionally, you can now train the model
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))