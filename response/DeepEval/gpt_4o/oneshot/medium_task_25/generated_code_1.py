import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Dense, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Path 1: Single 1x1 Convolution
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Path 2: Average Pooling followed by 1x1 Convolution
    path2 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)
    
    # Path 3: 1x1 Convolution followed by two parallel 1x3 and 3x1 Convolutions
    path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path3_1x3 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(path3)
    path3_3x1 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(path3)
    path3 = Concatenate()([path3_1x3, path3_3x1])
    
    # Path 4: 1x1 Convolution followed by 3x3 Convolution, then two parallel 1x3 and 3x1 Convolutions
    path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path4)
    path4_1x3 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(path4)
    path4_3x1 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(path4)
    path4 = Concatenate()([path4_1x3, path4_3x1])
    
    # Concatenate all paths
    multi_scale_features = Concatenate()([path1, path2, path3, path4])
    
    # Flatten the concatenated output
    flatten_layer = Flatten()(multi_scale_features)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model