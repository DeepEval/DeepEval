import keras
from keras.layers import Input, Conv2D, Concatenate, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 64))
    
    # 1x1 convolution for channel compression
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Parallel convolutional layers
    conv2_1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv2_3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Concatenate the outputs
    concat_layer = Concatenate()([conv2_1x1, conv2_3x3])
    
    # Flatten the output
    flatten_layer = Flatten()(concat_layer)
    
    # Fully connected layers
    dense1 = Dense(units=64, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model