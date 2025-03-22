import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Define three average pooling layers with different pool sizes and strides
    avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='same')(input_layer)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='same')(input_layer)
    
    # Flatten the outputs of each pooling layer
    flatten1 = Flatten()(avg_pool1)
    flatten2 = Flatten()(avg_pool2)
    flatten3 = Flatten()(avg_pool3)
    
    # Concatenate the flattened outputs
    concatenated = Concatenate()([flatten1, flatten2, flatten3])
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(concatenated)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model