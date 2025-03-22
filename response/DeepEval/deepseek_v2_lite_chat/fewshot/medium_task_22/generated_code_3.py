import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Concatenate, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: 3x3 convolutions
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(branch1)
    
    # Branch 2: 1x1 convolution, followed by two 3x3 convolutions
    branch2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(branch2)
    
    # Branch 3: Max pooling
    branch3 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(input_layer)
    
    # Concatenate the outputs from all branches
    concat = Concatenate(axis=-1)([branch1, branch2, branch3])
    
    # Flatten the concatenated features
    flat = Flatten()(concat)
    
    # Fully connected layers for classification
    dense1 = Dense(units=512, activation='relu')(flat)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model