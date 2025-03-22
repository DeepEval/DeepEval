import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Input layer for CIFAR-10 images (32x32x3)
    input_shape = (32, 32, 3)
    inputs = Input(shape=input_shape)
    
    # Initial 1x1 convolutional layer
    x = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(inputs)
    
    # Branch 1: 3x3 Convolutional Layer
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)

    # Branch 2: Max Pooling + 3x3 Convolutional Layer + UpSampling
    branch2 = MaxPooling2D(pool_size=(2, 2))(x)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(branch2)
    branch2 = UpSampling2D(size=(2, 2))(branch2)

    # Branch 3: Max Pooling + 3x3 Convolutional Layer + UpSampling
    branch3 = MaxPooling2D(pool_size=(2, 2))(x)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(branch3)
    branch3 = UpSampling2D(size=(2, 2))(branch3)

    # Concatenate the outputs of all branches
    concatenated = Concatenate()([branch1, branch2, branch3])
    
    # Final 1x1 convolutional layer
    x = Conv2D(filters=128, kernel_size=(1, 1), activation='relu', padding='same')(concatenated)

    # Flatten the output for the fully connected layers
    x = Flatten()(x)

    # Fully connected layers
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)  # 10 classes for CIFAR-10

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
model = dl_model()
model.summary()  # To display the model architecture