import keras
from keras.layers import Input, Conv2D, Flatten, Dense, Concatenate
from keras.models import Model

def dl_model():
    # Define the shape of the input data
    input_shape = (32, 32, 64)
    
    # Input layer
    input_layer = Input(shape=input_shape)
    
    # Step 1: Compressing the input channels with a 1x1 convolutional layer
    compressed = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # Step 2: Expand features using parallel convolutional layers
    
    # 1x1 Convolution branch
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(compressed)
    
    # 3x3 Convolution branch
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(compressed)
    
    # Step 3: Concatenating the results of the two branches
    concatenated = Concatenate()([branch1, branch2])
    
    # Step 4: Flattening the feature map into a one-dimensional vector
    flattened = Flatten()(concatenated)
    
    # Step 5: Passing through fully connected layers
    fc1 = Dense(units=128, activation='relu')(flattened)
    fc2 = Dense(units=10, activation='softmax')(fc1)  # Assuming 10 classes for classification
    
    # Defining the model
    model = Model(inputs=input_layer, outputs=fc2)
    
    return model

# To instantiate the model, you can call the function
model = dl_model()