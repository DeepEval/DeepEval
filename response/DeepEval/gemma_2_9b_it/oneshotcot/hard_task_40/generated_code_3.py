import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Dropout, BatchNormalization

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))

    # First Block: Average Pooling
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=1)(input_layer)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=2)(input_layer)
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=4)(input_layer)
    
    flattened_pools = [Flatten()(pool) for pool in [pool1, pool2, pool3]]
    concat_pools = Concatenate()(flattened_pools)
    reshape_layer = Reshape((1, -1))(concat_pools) 
    dense1 = Dense(units=64, activation='relu')(reshape_layer)

    # Second Block: Feature Extraction
    
    def feature_extraction_block(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path1 = Dropout(0.2)(path1)  

        path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
        path2 = Dropout(0.2)(path2) 

        path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Dropout(0.2)(path3)

        path4 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='same')(input_tensor)
        path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)
        path4 = Dropout(0.2)(path4)

        output_tensor = Concatenate(axis=3)([path1, path2, path3, path4])

        return output_tensor

    block_output = feature_extraction_block(dense1)

    
    dense2 = Dense(units=10, activation='softmax')(block_output)
    
    model = keras.Model(inputs=input_layer, outputs=dense2)
    return model