import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Global average pooling layer to capture global information from the feature map
    global_avg_pool = GlobalAveragePooling2D()(input_layer)
    
    # Two fully connected layers to generate weights whose size is the same as the channels of the input
    weights1 = Dense(units=64, activation='relu')(global_avg_pool)
    weights2 = Dense(units=32, activation='relu')(weights1)
    
    # Reshape the weights to align with the input shape
    weights = Reshape((1, 1, 32))(weights2)
    
    # Multiply the weights element-wise with the input feature map
    element_wise_product = Multiply()([input_layer, weights])
    
    # Flatten the result
    flatten_layer = Flatten()(element_wise_product)
    
    # Output layer to obtain the final probability distribution
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model