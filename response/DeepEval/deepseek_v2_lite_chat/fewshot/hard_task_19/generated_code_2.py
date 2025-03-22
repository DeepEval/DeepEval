import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    def main_path(x):
        # First convolutional layer
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
        # Second convolutional layer
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
        # Max pooling layer
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
        # Flatten layer
        flatten = Flatten()(pool1)
        return flatten
    
    # Branch path
    def branch_path(x):
        # Global average pooling layer
        gap = GlobalAveragePooling2D()(x)
        # Fully connected layer for channel weights
        fc1 = Dense(units=128, activation='relu')(gap)
        return fc1
    
    # Combine outputs from main path and branch path
    main_output = main_path(input_layer)
    branch_output = branch_path(input_layer)
    combined = keras.layers.concatenate([main_output, branch_output])
    
    # Fully connected layers for classification
    fc1 = Dense(units=512, activation='relu')(combined)
    output = Dense(units=10, activation='softmax')(fc1)
    
    # Model construction
    model = Model(inputs=input_layer, outputs=output)
    
    return model

# Instantiate the model
model = dl_model()

# Print model summary
model.summary()