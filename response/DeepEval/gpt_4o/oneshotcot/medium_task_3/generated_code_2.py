import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # First block
    conv1 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    
    # Second block
    conv2 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool1)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
    
    # Adjust the input to match the max pool output size for addition
    adjusted_input = MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(input_layer)
    
    # Add the output of the second block with the adjusted input
    added = Add()([max_pool2, adjusted_input])
    
    # Flatten and add dense layer for classification
    flatten_layer = Flatten()(added)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model