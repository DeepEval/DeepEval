import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, AveragePooling2D, Flatten, Dense

def dl_model():
    # Input Layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Path 1
    path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Path 2
    path2_1 = Conv2D(filters=64, kernel_size=(1, 7), padding='same', activation='relu')(input_layer)
    path2_2 = Conv2D(filters=64, kernel_size=(7, 1), padding='same', activation='relu')(input_layer)
    path2 = Concatenate(axis=-1)([path2_1, path2_2])
    
    # Path 3
    path3_1 = Conv2D(filters=64, kernel_size=(1, 7), padding='same', activation='relu')(input_layer)
    path3_2 = Conv2D(filters=64, kernel_size=(7, 1), padding='same', activation='relu')(input_layer)
    path3 = Concatenate(axis=-1)([path3_1, path3_2])
    
    # Path 4
    path4_1 = AveragePooling2D(pool_size=(1, 8), strides=(1, 1), padding='same')(input_layer)
    path4_2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(path4_1)
    path4 = Concatenate(axis=-1)([path4_1, path4_2])
    
    # Concatenate all paths
    concat_layer = Concatenate(axis=-1)([path1, path2, path3, path4])
    
    # Flatten the concatenated layer
    flatten_layer = Flatten()(concat_layer)
    
    # Fully connected layer
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense_layer)
    
    # Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model