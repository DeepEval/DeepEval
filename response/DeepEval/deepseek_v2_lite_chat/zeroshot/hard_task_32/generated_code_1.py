import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, Dropout

def dl_model():
    # Input shape
    input_shape = (28, 28, 1)
    
    # Branch 1: Depthwise Separable Convolution Layer
    branch1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_shape)
    branch1 = DepthwiseConv2D((3, 3), activation='relu', padding='same')(branch1)
    branch1 = Conv2D(32, (1, 1), activation='relu', padding='same')(branch1)
    branch1 = Dropout(0.5)(branch1)
    
    # Branch 2: Depthwise Separable Convolution Layer
    branch2 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_shape)
    branch2 = DepthwiseConv2D((3, 3), activation='relu', padding='same')(branch2)
    branch2 = Conv2D(64, (1, 1), activation='relu', padding='same')(branch2)
    branch2 = Dropout(0.5)(branch2)
    
    # Branch 3: Depthwise Separable Convolution Layer
    branch3 = Conv2D(128, (3, 3), activation='relu', padding='same')(input_shape)
    branch3 = DepthwiseConv2D((3, 3), activation='relu', padding='same')(branch3)
    branch3 = Conv2D(128, (1, 1), activation='relu', padding='same')(branch3)
    branch3 = Dropout(0.5)(branch3)
    
    # Concatenate all branches
    combined = concatenate([branch1, branch2, branch3])
    
    # Flatten and Fully Connected Layers
    combined = Flatten()(combined)
    combined = Dense(1024, activation='relu')(combined)
    combined = Dropout(0.5)(combined)
    
    output = Dense(10, activation='softmax')(combined)  # Assuming 10 classes (0-9)
    
    # Model
    model = Model(inputs=[branch1, branch2, branch3], outputs=output)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Create the model
model = dl_model()