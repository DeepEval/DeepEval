import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dense, Conv2D, Concatenate, Reshape

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    pool_1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    pool_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    pool_3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)
    
    flatten_pool_1 = Flatten()(pool_1)
    flatten_pool_2 = Flatten()(pool_2)
    flatten_pool_3 = Flatten()(pool_3)
    
    concat_pool = Concatenate()([flatten_pool_1, flatten_pool_2, flatten_pool_3])
    
    # Block 2
    path_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path_3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path_4 = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same')(input_layer)
    
    concat_path = Concatenate()([path_1, path_2, path_3, path_4])
    
    reshape_path = Reshape((4, -1))(concat_path)
    
    # Classification
    flatten_layer = Flatten()(reshape_path)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model