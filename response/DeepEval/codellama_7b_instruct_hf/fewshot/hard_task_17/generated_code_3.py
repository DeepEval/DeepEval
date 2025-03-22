import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Flatten, Add
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Global Average Pooling
    gavg = GlobalAveragePooling2D()(input_layer)
    fc1 = Dense(64, activation='relu')(gavg)
    fc2 = Dense(64, activation='relu')(fc1)
    fc3 = Dense(64, activation='relu')(fc2)
    weights = Reshape(target_shape=(32, 32, 3))(fc3)
    
    # Block 2: Convolutional and Max Pooling
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    conv2 = Conv2D(32, (3, 3), activation='relu')(conv1)
    mp = MaxPooling2D((2, 2))(conv2)
    
    # Add branch from Block 1 to the output of Block 2
    branch = Add()([mp, weights])
    
    # Flatten and output
    flattened = Flatten()(branch)
    output = Dense(10, activation='softmax')(flattened)
    
    # Create model
    model = Model(inputs=input_layer, outputs=output)
    
    return model