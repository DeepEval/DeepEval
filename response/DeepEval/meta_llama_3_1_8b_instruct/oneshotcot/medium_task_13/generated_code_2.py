import keras
from keras.layers import Input, Conv2D, Concatenate, BatchNormalization, Flatten, Dense, MaxPooling2D

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Convolutional layer 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    # Convolutional layer 2
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    # Convolutional layer 3
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    
    # Concatenate the output from each layer along the channel dimension for the next layer
    concat_output = Concatenate()([conv1, conv2, conv3])
    
    # Batch normalization
    bath_norm = BatchNormalization()(concat_output)
    
    # Flatten the output
    flatten_layer = Flatten()(bath_norm)
    
    # Dense layer 1
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    # Dense layer 2
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model