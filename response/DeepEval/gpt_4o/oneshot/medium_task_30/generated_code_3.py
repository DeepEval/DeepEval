import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Average Pooling Layers
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)
    
    # Flatten each pool's output
    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat3 = Flatten()(pool3)
    
    # Concatenate the flattened outputs
    concatenated = Concatenate()([flat1, flat2, flat3])
    
    # Further processing through fully connected layers
    dense1 = Dense(units=256, activation='relu')(concatenated)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model