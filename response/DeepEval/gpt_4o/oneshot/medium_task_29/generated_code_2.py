import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First max pooling layer with 1x1 pool size
    max_pooling_1x1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    flatten_1x1 = Flatten()(max_pooling_1x1)
    
    # Second max pooling layer with 2x2 pool size
    max_pooling_2x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    flatten_2x2 = Flatten()(max_pooling_2x2)
    
    # Third max pooling layer with 4x4 pool size
    max_pooling_4x4 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)
    flatten_4x4 = Flatten()(max_pooling_4x4)
    
    # Concatenate the flattened outputs
    concatenated_features = Concatenate()([flatten_1x1, flatten_2x2, flatten_4x4])
    
    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(concatenated_features)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model