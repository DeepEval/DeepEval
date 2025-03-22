import keras
from keras.layers import Input, Lambda, Reshape, Permute, Conv2D, Dense

def dl_model():     
    input_layer = Input(shape=(32, 32, 3))
    
    # Reshape and permute input to enable channel shuffling
    reshaped_layer = Lambda(lambda x: keras.backend.reshape(x, (-1, 32, 32, 3/3, 3)))(input_layer)
    permuted_layer = Lambda(lambda x: keras.backend.permute_dimensions(x, [0, 1, 2, 4, 3]))(reshaped_layer)
    reshaped_back_layer = Lambda(lambda x: keras.backend.reshape(x, (-1, 32, 32, 3)))(permuted_layer)
    
    # Apply fully connected layer with softmax activation for classification
    output_layer = Dense(units=10, activation='softmax')(reshaped_back_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model