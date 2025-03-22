import keras
from keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten, Concatenate, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First branch: 1x1 convolution
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # Second branch: 1x1 convolution -> 3x3 convolution
    branch2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(branch2)
    
    # Third branch: 1x1 convolution -> 5x5 convolution
    branch3 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)
    branch3 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(branch3)
    
    # Fourth branch: 3x3 max pooling -> 1x1 convolution
    branch4 = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(input_layer)
    branch4 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(branch4)
    
    # Concatenate the outputs of the four branches
    concatenated = Concatenate()([branch1, branch2, branch3, branch4])
    
    # Flatten the concatenated features
    flattened = Flatten()(concatenated)
    
    # Two fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=dense2)
    
    return model