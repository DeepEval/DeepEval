import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    max_pooling1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    flatten1 = Flatten()(max_pooling1)
    
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    flatten2 = Flatten()(max_pooling2)
    
    max_pooling3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)
    flatten3 = Flatten()(max_pooling3)
    
    # Concatenate the flattened output from each pooling layer
    output_tensor = Concatenate()([flatten1, flatten2, flatten3])
    
    batch_norm = BatchNormalization()(output_tensor)
    
    dense1 = Dense(units=128, activation='relu')(batch_norm)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model