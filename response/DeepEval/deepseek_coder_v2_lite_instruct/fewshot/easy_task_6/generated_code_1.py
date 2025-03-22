import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Main path
    conv1_main = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2_main = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_main)
    
    # Branch path
    conv1_branch = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Addition operation
    added = Add()([conv2_main, conv1_branch])
    
    # Flatten layer
    flattened = Flatten()(added)
    
    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flattened)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model