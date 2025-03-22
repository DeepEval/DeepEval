import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Concatenate, Dense, Reshape
from keras.layers import AveragePooling2D

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # Processing block 1
    avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_layer)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(avg_pool1)
    avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='valid')(avg_pool2)
    
    flat_output = Flatten()(Concatenate()([avg_pool3]))
    
    reshape_layer = Reshape((-1, flat_output.shape[1]*flat_output.shape[2]*flat_output.shape[3]))(flat_output)
    
    # Processing block 2
    def block(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1)
        path3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
        path4 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='valid')(path3)
        
        dropout1 = keras.layers.Dropout(0.5)(path4)
        concat_tensor = Concatenate()([dropout1])

        return concat_tensor
    
    block_output = block(reshape_layer)
    dense1 = Dense(units=128, activation='relu')(block_output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model