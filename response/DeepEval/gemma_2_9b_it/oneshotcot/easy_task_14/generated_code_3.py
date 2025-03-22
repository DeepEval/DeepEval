import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    global_average_pooling = GlobalAveragePooling2D()(input_layer)

    dense1 = Dense(units=32, activation='relu')(global_average_pooling)
    dense2 = Dense(units=32, activation='relu')(dense1)
    
    weights = Dense(units=32, activation='relu')(dense2)  
    weights = Reshape((32, 32, 3))(weights) # Reshape to match input shape
    
    elementwise_product = Multiply()([input_layer, weights])
    
    flatten_layer = Flatten()(elementwise_product)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model