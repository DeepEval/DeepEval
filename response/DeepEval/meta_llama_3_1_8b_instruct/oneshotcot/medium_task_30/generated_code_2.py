import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Average pooling layer 1
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_layer)
    
    # Average pooling layer 2
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(pool1)
    
    # Average pooling layer 3
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='valid')(pool2)
    
    # Concatenate the outputs of pooling layers
    output_tensor = Concatenate()([pool1, pool2, pool3])
    
    # Flatten the concatenated output
    flatten_layer = Flatten()(output_tensor)
    
    # First dense layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    # Second dense layer
    dense2 = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model