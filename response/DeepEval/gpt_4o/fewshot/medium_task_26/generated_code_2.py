import keras
from keras.layers import Input, Conv2D, Concatenate, Flatten, Dense

def dl_model():
    # Define the input layer with the given shape
    input_layer = Input(shape=(32, 32, 64))
    
    # Channel compression using a 1x1 convolutional layer
    compressed = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Parallel 1x1 and 3x3 convolutions for feature expansion
    conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(compressed)
    conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(compressed)
    
    # Concatenate the outputs of the parallel convolution layers
    concatenated = Concatenate()([conv1x1, conv3x3])
    
    # Flatten the concatenated feature map into a one-dimensional vector
    flatten_layer = Flatten()(concatenated)
    
    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model