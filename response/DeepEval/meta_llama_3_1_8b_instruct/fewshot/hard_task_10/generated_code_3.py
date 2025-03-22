import keras
from keras.layers import Input, Conv2D, Add, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Feature extraction paths
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path2)
    path2 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path2)
    
    # Concatenate the outputs from the two paths and apply a 1x1 convolution to align the output dimensions
    output = Concatenate()([path1, path2])
    output = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(output)
    
    # Add a branch that connects directly to the input and merge the outputs of the main path and the branch
    branch = Add()([input_layer, output])
    
    # Flatten the merged output and apply two fully connected layers for classification
    flatten = Flatten()(branch)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model