import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Multiply, Reshape, Flatten

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Global Average Pooling to capture global information from the feature map
    gap = GlobalAveragePooling2D()(input_layer)
    
    # Fully connected layers to generate weights
    dense1 = Dense(units=32, activation='relu')(gap)
    dense2 = Dense(units=3, activation='sigmoid')(dense1)  # Output size should match the number of channels
    
    # Reshape the weights to align with input shape
    reshape_layer = Reshape((1, 1, 3))(dense2)
    
    # Multiply element-wise with the input feature map
    scaled_input = Multiply()([input_layer, reshape_layer])
    
    # Flatten the result
    flatten_layer = Flatten()(scaled_input)
    
    # Final fully connected layer to produce the output
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model