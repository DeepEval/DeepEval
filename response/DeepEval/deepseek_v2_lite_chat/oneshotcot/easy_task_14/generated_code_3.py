import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Compress the input features with global average pooling
    conv_layer = GlobalAveragePooling2D()(input_layer)
    
    # Fully connected layers to generate weights
    dense1 = Dense(units=512, activation='relu')(conv_layer)
    dense2 = Dense(units=256, activation='relu')(dense1)
    
    # Reshape weights to align with the input shape and multiply element-wise
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model