import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    def block(input_tensor):
        # First layer of the block
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        x = BatchNormalization()(x)
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        
        # Second layer of the block
        y = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        y = BatchNormalization()(y)
        y = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(y)
        y = BatchNormalization()(y)
        
        # Third layer of the block
        z = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        z = BatchNormalization()(z)
        z = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(z)
        z = BatchNormalization()(z)
        
        # Concatenate the outputs of the three paths
        concatenated = Concatenate()([x, y, z])
        
        return concatenated
    
    # First pathway
    pathway1 = block(input_tensor=input_layer)
    
    # Second pathway
    pathway2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    pathway2 = block(input_tensor=pathway2)
    
    # Concatenate the outputs of both pathways
    concatenated_pathways = Concatenate()([pathway1, pathway2])
    
    # Flatten the concatenated pathways
    flatten_layer = Flatten()(concatenated_pathways)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense1 = Dropout(0.3)(dense1)
    dense2 = Dense(units=64, activation='relu')(dense1)
    dense2 = Dropout(0.3)(dense2)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model