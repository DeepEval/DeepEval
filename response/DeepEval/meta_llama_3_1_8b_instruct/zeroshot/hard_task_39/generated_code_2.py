# Import necessary packages
from tensorflow.keras.layers import Input, MaxPooling2D, Flatten, Concatenate, Reshape, Dense, Conv2D, MaxPooling
from tensorflow.keras.models import Model

def dl_model():
    # Define input shape
    input_shape = (28, 28, 1)
    
    # Define inputs
    inputs = Input(shape=input_shape)
    
    # Block 1: Max pooling layers
    x = inputs
    x = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(x)
    
    # Flatten the pooling results
    x = Flatten()(x)
    
    # Concatenate the flattened vectors
    x = Concatenate()([x, x, x])
    
    # Reshape to 4-dimensional tensor
    x = Reshape((3 * 784 // 9, 9))(x)
    
    # Block 2: Convolutional and max pooling layers
    branch1 = Conv2D(32, (1, 1), activation='relu')(x)
    branch1 = MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(branch1)
    branch1 = Flatten()(branch1)
    
    branch2 = Conv2D(32, (3, 3), activation='relu')(x)
    branch2 = MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(branch2)
    branch2 = Flatten()(branch2)
    
    branch3 = Conv2D(32, (5, 5), activation='relu')(x)
    branch3 = MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(branch3)
    branch3 = Flatten()(branch3)
    
    # Concatenate the outputs from all branches
    x = Concatenate()([branch1, branch2, branch3])
    
    # Classification layer
    x = Dense(10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=x)
    
    return model

# Test the model
model = dl_model()
model.summary()