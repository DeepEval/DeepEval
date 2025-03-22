import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten

def dl_model():  
    
    input_layer = Input(shape=(32, 32, 3))  

    # Global average pooling
    pooling_layer = GlobalAveragePooling2D()(input_layer)

    # Fully connected layers for channel correlation learning
    dense1 = Dense(units=3 * 32 * 32)(pooling_layer)  
    dense2 = Dense(units=3 * 32 * 32)(dense1)

    # Reshape weights to align with input shape
    reshaped_weights = Reshape((32, 32, 3))(dense2)

    # Element-wise multiplication with input feature map
    weighted_input = Multiply()([input_layer, reshaped_weights])

    # Flatten the result and final fully connected layer
    flattened_output = Flatten()(weighted_input)
    output_layer = Dense(units=10, activation='softmax')(flattened_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model