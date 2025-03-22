import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    avg_pooling1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='valid')(conv)
    avg_pooling2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)
    avg_pooling3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='valid')(conv)
    
    def block1(input_tensor):
        path1 = Flatten()(AveragePooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_tensor))
        path2 = Flatten()(AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_tensor))
        path3 = Flatten()(AveragePooling2D(pool_size=(4, 4), strides=4, padding='valid')(input_tensor))
        output_tensor = Concatenate()([path1, path2, path3])
        output_tensor = Dropout(0.2)(output_tensor)
        return output_tensor
    
    block1_output = block1(input_tensor=conv)
    
    dense = Dense(units=128, activation='relu')(block1_output)
    reshaped = keras.layers.Reshape((4,))(dense)
    
    def block2(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(keras.layers.Reshape((1, 1, -1))(input_tensor))
        path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(keras.layers.Reshape((2, 2, -1))(input_tensor))
        path3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(keras.layers.Reshape((2, 2, -1))(input_tensor))
        path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(keras.layers.Reshape((3, 3, -1))(input_tensor))
        path5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(keras.layers.Reshape((3, 3, -1))(input_tensor))
        path6 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(keras.layers.Reshape((4, 4, -1))(input_tensor))
        path7 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        path7 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(keras.layers.Reshape((2, 2, -1))(path7))
        output_tensor = Concatenate()([path1, path2, path3, path4, path5, path6, path7])
        return output_tensor
    
    block2_output = block2(input_tensor=reshaped)
    batch_norm = BatchNormalization()(block2_output)
    flatten_layer = Flatten()(batch_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model