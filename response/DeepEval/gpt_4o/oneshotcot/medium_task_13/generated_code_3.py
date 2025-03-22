import keras
from keras.layers import Input, Conv2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    
    # Concatenation of the input and the first conv layer
    concat1 = Concatenate(axis=-1)([input_layer, conv1])
    
    # Second convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(concat1)
    
    # Concatenation of the output from first concat and the second conv layer
    concat2 = Concatenate(axis=-1)([concat1, conv2])
    
    # Third convolutional layer
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(concat2)
    
    # Concatenation of the output from second concat and the third conv layer
    concat3 = Concatenate(axis=-1)([concat2, conv3])
    
    # Flatten the result
    flatten_layer = Flatten()(concat3)
    
    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model