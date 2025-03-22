import keras
from keras.layers import Input, AveragePooling2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Average pooling layer with 1x1 window and stride
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_layer)
    
    # Average pooling layer with 2x2 window and stride
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(pool1)
    
    # Average pooling layer with 4x4 window and stride
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='valid')(pool2)
    
    # Concatenate the outputs of the pooling layers
    concat_output = Concatenate()([pool1, pool2, pool3])
    
    # Flatten the concatenated output
    flatten_layer = Flatten()(concat_output)
    
    # Dense layer with 128 units and ReLU activation
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    # Dense layer with 64 units and ReLU activation
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output layer with 10 units and softmax activation for classification
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Create the Keras model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model