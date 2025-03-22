import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add, AveragePooling2D

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    def basic_block(input_tensor):

        path = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path = BatchNormalization()(path)
        path = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path)
        path = BatchNormalization()(path)

        output_tensor = Add()([path, input_tensor])

        return output_tensor
    
    res_block1 = basic_block(input_tensor=conv1)
    res_block2 = basic_block(input_tensor=res_block1)

    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(res_block2)
    
    def residual_block(input_tensor):

        path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path = BatchNormalization()(path)
        path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path)
        path = BatchNormalization()(path)

        branch = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)

        output_tensor = Add()([path, branch])

        return output_tensor
    
    res_block3 = residual_block(input_tensor=conv2)
    res_block4 = residual_block(input_tensor=res_block3)

    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(res_block4)

    conv4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv3)

    avg_pool = AveragePooling2D(pool_size=(8, 8), strides=1, padding='valid')(conv4)

    flatten_layer = Flatten()(avg_pool)
    dense1 = Dense(units=64, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model