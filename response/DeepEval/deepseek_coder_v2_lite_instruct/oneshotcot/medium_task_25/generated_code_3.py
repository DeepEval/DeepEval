import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Path 1: Single 1x1 convolution
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Path 2: Average pooling followed by a 1x1 convolution
    path2 = AveragePooling2D(pool_size=(2, 2), strides=2)(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)
    
    # Path 3: 1x1 convolution followed by two parallel 1x3 and 3x1 convolutions
    path3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path3_1 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(path3)
    path3_2 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(path3)
    path3 = Concatenate()([path3_1, path3_2])
    
    # Path 4: 1x1 convolution followed by a 3x3 convolution, then followed by two parallel 1x3 and 3x1 convolutions
    path4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path4 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path4)
    path4_1 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(path4)
    path4_2 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(path4)
    path4 = Concatenate()([path4_1, path4_2])
    
    # Concatenate the outputs of these paths
    multi_scale_feature = Concatenate()([path1, path2, path3, path4])
    
    # Batch normalization and flatten the result
    batch_norm = BatchNormalization()(multi_scale_feature)
    flatten_layer = Flatten()(batch_norm)
    
    # Fully connected layers for classification
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model