import keras
from keras.layers import Input, AveragePooling2D, Concatenate, Dense, Flatten, Reshape, Conv2D, Dropout

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # First block
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=1)(input_layer)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=2)(input_layer)
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=4)(input_layer)
    
    flatten_layer = Flatten()(Concatenate()([pool1, pool2, pool3]))
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    reshape_layer = Reshape((1, 1, 256))(dense1)
    
    # Second block
    path1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(reshape_layer)
    
    path2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(reshape_layer)
    path2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(path2)
    path2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(path2)
    
    path3 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(reshape_layer)
    path3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(path3)
    
    path4 = AveragePooling2D(pool_size=(1, 1), strides=1)(reshape_layer)
    path4 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(path4)
    
    concat_layer = Concatenate(axis=3)([path1, path2, path3, path4])
    
    dropout_layer = Dropout(0.5)(concat_layer)
    
    flatten_layer_2 = Flatten()(dropout_layer)
    dense2 = Dense(units=128, activation='relu')(flatten_layer_2)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model