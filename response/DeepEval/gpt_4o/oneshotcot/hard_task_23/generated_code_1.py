import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Conv2DTranspose, Concatenate, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial 1x1 convolutional layer
    initial_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Branch 1: Local Feature Extraction with two 3x3 Convolutional layers
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(initial_conv)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)
    
    # Branch 2: Average Pooling, 3x3 Convolution, and Transposed Convolution
    branch2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(initial_conv)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(branch2)
    
    # Branch 3: Average Pooling, 3x3 Convolution, and Transposed Convolution
    branch3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(initial_conv)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(branch3)
    
    # Concatenate the outputs of the three branches
    concatenated = Concatenate()([branch1, branch2, branch3])
    
    # Refinement with a 1x1 convolutional layer
    refined = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)
    
    # Flatten and fully connected layer for classification
    flatten_layer = Flatten()(refined)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model