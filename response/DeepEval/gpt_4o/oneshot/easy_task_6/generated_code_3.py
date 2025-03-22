import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Main path
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Branch path
    branch = input_layer  # Direct connection to the input
    
    # Combining paths through addition
    combined = Add()([conv2, branch])
    
    # Final layers
    flatten_layer = Flatten()(combined)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model