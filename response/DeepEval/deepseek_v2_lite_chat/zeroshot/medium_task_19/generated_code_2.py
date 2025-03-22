import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from keras.applications.vgg16 import VGG16

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First branch: 1x1 convolution for dimensionality reduction
    branch1 = Conv2D(filters=64, kernel_size=1, activation='relu')(input_layer)
    
    # Second branch: 1x1 convolution -> 3x3 convolution
    branch2 = Conv2D(filters=64, kernel_size=1, activation='relu')(input_layer)
    branch2 = Conv2D(filters=64, kernel_size=3, activation='relu')(branch2)
    
    # Third branch: 1x1 convolution -> 5x5 convolution
    branch3 = Conv2D(filters=128, kernel_size=1, activation='relu')(input_layer)
    branch3 = Conv2D(filters=128, kernel_size=5, activation='relu')(branch3)
    
    # Fourth branch: 3x3 max pooling -> 1x1 convolution
    branch4 = MaxPooling2D(pool_size=(3, 3))(input_layer)
    branch4 = Conv2D(filters=256, kernel_size=1, activation='relu')(branch4)
    
    # Concatenate the outputs of all branches
    concat = concatenate([branch1, branch2, branch3, branch4])
    
    # Flatten the concatenated features
    flat = Flatten()(concat)
    
    # Fully connected layers for classification
    output_layer = Dense(units=10, activation='softmax')(flat)
    
    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Example usage:
model = dl_model()
model.summary()