import keras
from keras.layers import Conv2D, Add, Flatten, Dense, Input
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Branch 1
    branch1 = Conv2D(32, kernel_size=(1, 1), activation='relu')(input_layer)
    branch1 = Conv2D(32, kernel_size=(3, 3), activation='relu', depthwise=True)(branch1)
    branch1 = Conv2D(64, kernel_size=(1, 1), activation='relu')(branch1)
    
    # Branch 2
    branch2 = Conv2D(32, kernel_size=(1, 1), activation='relu')(input_layer)
    branch2 = Conv2D(32, kernel_size=(3, 3), activation='relu', depthwise=True)(branch2)
    branch2 = Conv2D(64, kernel_size=(1, 1), activation='relu')(branch2)
    
    # Branch 3
    branch3 = Conv2D(32, kernel_size=(1, 1), activation='relu')(input_layer)
    branch3 = Conv2D(32, kernel_size=(3, 3), activation='relu', depthwise=True)(branch3)
    branch3 = Conv2D(64, kernel_size=(1, 1), activation='relu')(branch3)
    
    # Concatenate branches
    concatenated = Add()([branch1, branch2, branch3])
    
    # Flatten and fully connected layers
    flattened = Flatten()(concatenated)
    output_layer = Dense(10, activation='softmax')(flattened)  # Assuming 10 classes
    
    # Model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Get the constructed model
model = dl_model()
model.summary()