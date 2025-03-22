import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Path 1: Average Pooling
    path1 = AveragePooling2D(pool_size=(2, 2))(input_layer)
    
    # Path 2: 1x1 Convolution
    path2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # Path 3: 1x3 and 3x1 Convolution
    path3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    path3_1x3 = Conv2D(filters=32, kernel_size=(1, 3), padding='same')(path3)
    path3_3x1 = Conv2D(filters=32, kernel_size=(3, 1), padding='same')(path3)
    path3 = Concatenate(axis=-1)([path3_1x3, path3_3x1])
    
    # Path 4: 1x1 Convolution, then 3x3 Convolution
    path4 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)
    path4 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(path4)
    
    # Concatenate all paths
    concat_layer = Concatenate(axis=-1)([path1, path2, path3, path4])
    
    # Batch Normalization and Flatten
    batch_norm = BatchNormalization()(concat_layer)
    flatten = Flatten()(batch_norm)
    
    # Fully Connected Layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model