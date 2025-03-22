import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Path 1: 1x1 convolution
    path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Path 2: 3x3 convolution, followed by 1x1 convolution
    path2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(path2)
    
    # Path 3: 3x3 convolution, followed by 1x1 convolution
    path3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(path3)
    
    # Path 4: Max pooling, followed by 1x1 convolution
    path4 = MaxPooling2D(pool_size=(2, 2))(input_layer)
    path4 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(path4)
    
    # Concatenate the paths
    concat = Concatenate(axis=-1)([path1, path2, path3, path4])
    
    # Batch normalization and flattening
    bn = BatchNormalization()(concat)
    flatten = Flatten()(bn)
    
    # Dense layer
    dense1 = Dense(units=128, activation='relu')(flatten)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model