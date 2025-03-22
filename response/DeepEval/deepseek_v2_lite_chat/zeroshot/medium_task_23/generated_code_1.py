from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 pixels
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Branch 1: Single 1x1 convolution
    branch1 = Conv2D(32, (1, 1), activation='relu')(inputs)
    
    # Branch 2: 1x1 convolution followed by 1x7 and 7x1 convolutions
    branch2 = Conv2D(32, (1, 1), padding='same')(inputs)
    branch2 = Conv2D(32, (7, 1), padding='valid')(branch2)
    branch2 = Conv2D(32, (1, 1), padding='same')(branch2)
    
    # Branch 3: 1x1 convolution followed by two sets of 1x7 and 7x1 convolutions
    branch3 = Conv2D(32, (1, 1), padding='same')(inputs)
    branch3 = Conv2D(32, (7, 1), padding='valid')(branch3)
    branch3 = Conv2D(32, (7, 1), padding='valid')(branch3)
    branch3 = Conv2D(32, (1, 1), padding='same')(branch3)
    
    # Branch 4: Average pooling followed by a 1x1 convolution
    branch4 = MaxPooling2D(pool_size=(4, 4))(inputs)
    branch4 = Conv2D(32, (1, 1), padding='same')(branch4)
    
    # Concatenate the outputs of all branches
    concat = concatenate([branch1, branch2, branch3, branch4])
    
    # Flatten and pass through a fully connected layer
    flat = Flatten()(concat)
    output = Dense(10, activation='softmax')(flat)  # Assuming 10 classes for CIFAR-10
    
    # Create the model
    model = Model(inputs=inputs, outputs=output)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Example usage:
# model = dl_model()
# model.summary()