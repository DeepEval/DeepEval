import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, concatenate, Dense, GlobalAveragePooling2D
from keras.layers import Layer

# Constants
IMG_ROWS, IMG_COLS, IMG_CHANNELS = 32, 32, 3
NUM_CLASSES = 10

def dl_model():
    # Input layer
    input_layer = Input(shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS))
    
    # First branch for dimensionality reduction
    branch1 = Conv2D(16, (1, 1), activation='relu')(input_layer)
    
    # Second branch for feature extraction
    branch2 = Conv2D(32, (1, 1), activation='relu')(branch1)
    branch2 = Conv2D(32, (3, 3), activation='relu')(branch2)
    
    # Third branch for capturing spatial information
    branch3 = Conv2D(64, (1, 1), activation='relu')(input_layer)
    branch3 = Conv2D(64, (5, 5), activation='relu')(branch3)
    
    # Fourth branch for downsampling
    branch4 = Conv2D(64, (1, 1), activation='relu')(input_layer)
    branch4 = MaxPooling2D((3, 3), strides=(3, 3))(branch4)
    branch4 = Conv2D(64, (1, 1), activation='relu')(branch4)
    
    # Concatenate features from all branches
    concat = concatenate([branch2, branch3, branch4])
    
    # Flatten and pass through fully connected layers
    flat = Flatten()(concat)
    output = Dense(NUM_CLASSES, activation='softmax')(flat)
    
    # Model
    model = Model(inputs=input_layer, outputs=output)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# If running this code as a script
if __name__ == "__main__":
    model = dl_model()
    print(model.summary())