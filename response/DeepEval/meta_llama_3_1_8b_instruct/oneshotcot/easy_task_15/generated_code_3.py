import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Dropout, GlobalAveragePooling2D, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    max_pooling = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)

    def block(input_tensor):

        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='same')(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3, path4])

        return output_tensor
        
    block_output = block(input_tensor=max_pooling)
    dropout = Dropout(0.2)(block_output)
    block_output = block(dropout)
    dropout = Dropout(0.2)(block_output)

    global_avg_pool = GlobalAveragePooling2D()(dropout)
    flatten_layer = Flatten()(global_avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model