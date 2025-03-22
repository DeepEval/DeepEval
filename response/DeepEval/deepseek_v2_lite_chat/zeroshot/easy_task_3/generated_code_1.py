from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from keras.layers import Layer

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)  # Adjust if your input is different
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Block 1: Two Convolutional Layers, Max Pooling
    conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
    conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(conv1)
    maxpool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Block 2: Two Convolutional Layers, Max Pooling
    conv3 = Conv2D(64, kernel_size=(3, 3), activation='relu')(maxpool1)
    conv4 = Conv2D(128, kernel_size=(3, 3), activation='relu')(conv3)
    maxpool2 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # Flatten the feature maps
    flat = Flatten()(maxpool2)
    
    # Fully connected layers
    dense1 = Dense(128, activation='relu')(flat)
    dense2 = Dense(64, activation='relu')(dense1)
    output = Dense(10, activation='softmax')(dense2)  # Assuming 10 classes for MNIST
    
    # Create the model
    model = Model(inputs=inputs, outputs=output)
    
    return model

# Create the model
model = dl_model()
model.summary()