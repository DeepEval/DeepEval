import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 64))
    
    # Compression with 1x1 convolutional layer
    compressed = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # Expansion through parallel convolutional layers
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(compressed)
    conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(compressed)
    
    # Concatenate the results
    expanded = Concatenate()([conv1x1, conv3x3])
    
    # Flatten the output feature map
    flattened = Flatten()(expanded)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flattened)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model