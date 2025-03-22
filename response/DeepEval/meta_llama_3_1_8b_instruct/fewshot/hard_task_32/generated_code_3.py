import keras
from keras.layers import Input, DepthwiseConv2D, Conv2D, Concatenate, Dropout, Dense

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    
    def block(input_tensor):
        conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv)
        conv = Dropout(0.2)(conv)  # Add dropout to mitigate overfitting
        conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv)
        conv = Dropout(0.2)(conv)  # Add dropout to mitigate overfitting
        return conv
    
    branch1 = block(input_layer)
    branch2 = block(input_layer)
    branch3 = block(input_layer)
    
    output_tensor = Concatenate()([branch1, branch2, branch3])
    
    flatten_layer = Flatten()(output_tensor)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=dense2)
    
    return model