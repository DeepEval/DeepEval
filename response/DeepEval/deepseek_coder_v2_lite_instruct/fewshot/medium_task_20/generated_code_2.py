import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First path: 1x1 convolution
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Second path: Two 3x3 convolutions stacked after a 1x1 convolution
    path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
    path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
    
    # Third path: Single 3x3 convolution following a 1x1 convolution
    path3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path3)
    
    # Fourth path: Max pooling followed by a 1x1 convolution
    path4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    path4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)
    
    # Concatenate the outputs from these paths
    concatenated = Concatenate()([path1, path2, path3, path4])
    
    # Flatten the concatenated output
    flattened = Flatten()(concatenated)
    
    # Pass through a dense layer with 128 units
    dense = Dense(units=128, activation='relu')(flattened)
    
    # Final output layer with softmax activation
    output_layer = Dense(units=10, activation='softmax')(dense)
    
    # Construct and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model