import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Conv2DTranspose, Concatenate, Activation, Multiply

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer to adjust channels
    conv_init = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    
    # Block 1: Channel attention
    def block_1(input_tensor):
        path_1 = GlobalAveragePooling2D()(input_tensor)
        path_1 = Dense(units=32, activation='relu')(path_1)
        path_1 = Dense(units=input_tensor.shape[-1], activation='sigmoid')(path_1)
        path_2 = GlobalMaxPooling2D()(input_tensor)
        path_2 = Dense(units=32, activation='relu')(path_2)
        path_2 = Dense(units=input_tensor.shape[-1], activation='sigmoid')(path_2)
        attention = Add()([path_1, path_2])
        attention = Multiply()([input_tensor, attention])
        return attention
    
    # Block 2: Spatial attention
    def block_2(input_tensor):
        path_1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        path_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(path_1)
        path_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        path_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(path_2)
        concat = Concatenate(axis=-1)([path_1, path_2])
        concat = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same')(concat)
        concat = Activation('sigmoid')(concat)
        concat = Multiply()([input_tensor, concat])
        return concat
    
    # Main path
    path_1 = block_1(conv_init)
    path_2 = block_2(conv_init)
    main_path = Add()([path_1, path_2])
    main_path = Activation('relu')(main_path)
    
    # Final classification
    flatten = Flatten()(main_path)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model