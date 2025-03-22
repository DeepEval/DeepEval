import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    # Define the input layer with the shape of CIFAR-10 images
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: 3x3 Convolution
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Branch 2: 1x1 Convolution followed by two 3x3 Convolutions
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    
    # Branch 3: Max Pooling
    branch3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    
    # Concatenate the outputs from all branches
    concatenated_branches = Concatenate()([branch1, branch2, branch3])
    
    # Flatten the concatenated outputs
    flatten_layer = Flatten()(concatenated_branches)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Create and return the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model