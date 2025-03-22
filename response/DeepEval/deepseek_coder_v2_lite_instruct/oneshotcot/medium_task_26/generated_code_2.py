import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 64))
    
    # Compress the input channels with a 1x1 convolutional layer
    compressed = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Expand the features through two parallel convolutional layers
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(compressed)
    path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(compressed)
    
    # Concatenate the results
    concatenated = Concatenate()([path1, path2])
    
    # Apply batch normalization
    batch_norm = BatchNormalization()(concatenated)
    
    # Flatten the result
    flattened = Flatten()(batch_norm)
    
    # Pass through two fully connected layers
    dense1 = Dense(units=128, activation='relu')(flattened)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model