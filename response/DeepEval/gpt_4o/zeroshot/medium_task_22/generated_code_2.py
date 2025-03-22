from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from keras.models import Model
from keras.datasets import cifar10
from keras.utils import to_categorical

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First branch with a 3x3 convolution
    branch1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    
    # Second branch with 1x1 convolution followed by two 3x3 convolutions
    branch2 = Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)
    branch2 = Conv2D(32, (3, 3), activation='relu', padding='same')(branch2)
    branch2 = Conv2D(32, (3, 3), activation='relu', padding='same')(branch2)
    
    # Third branch with max pooling
    branch3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    
    # Concatenate the outputs from the three branches
    concatenated = Concatenate()([branch1, branch2, branch3])
    
    # Flatten the concatenated output
    flat = Flatten()(concatenated)
    
    # Fully connected layers for classification
    dense1 = Dense(128, activation='relu')(flat)
    dense2 = Dense(64, activation='relu')(dense1)
    
    # Output layer with 10 classes for CIFAR-10
    output_layer = Dense(10, activation='softmax')(dense2)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Example usage
model = dl_model()
model.summary()