import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense, AveragePooling2D

def dl_model():

    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Parallel convolutional paths
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path4 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(input_layer)
    
    # Concatenate outputs from the paths
    concatenated_output = Concatenate()([path1, path2, path3, path4])
    
    # Flatten the concatenated output and pass through dense layers
    flattened = Flatten()(concatenated_output)
    dense1 = Dense(units=128, activation='relu')(flattened)
    
    # Output layer with softmax activation
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model