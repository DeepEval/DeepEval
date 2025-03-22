import keras
from keras.layers import Input, Dense, GlobalAveragePooling2D, Flatten, Concatenate
from keras.models import Model

def dl_model():
    # Define the input layer with shape (32, 32, 3) for CIFAR-10 images
    input_layer = Input(shape=(32, 32, 3))
    
    # Define the first branch
    branch1 = input_layer
    branch1 = GlobalAveragePooling2D()(branch1)
    branch1 = Dense(64, activation='relu')(branch1)
    branch1 = Dense(64, activation='relu')(branch1)
    
    # Define the second branch
    branch2 = input_layer
    branch2 = GlobalAveragePooling2D()(branch2)
    branch2 = Dense(64, activation='relu')(branch2)
    branch2 = Dense(64, activation='relu')(branch2)
    
    # Concatenate the outputs from both branches
    merged = Concatenate()([branch1, branch2])
    
    # Flatten the merged output
    merged = Flatten()(merged)
    
    # Add a fully connected layer to produce the final classification results
    merged = Dense(10, activation='softmax')(merged)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=merged)
    
    return model