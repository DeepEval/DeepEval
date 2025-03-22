import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    max_pooling = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)

    def block(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        maxpool = AveragePooling2D(pool_size=(1, 1), strides=1, padding='same')(input_tensor)
        output_tensor = Concatenate()([conv1, conv2, conv3, maxpool])

        return output_tensor
    
    block_output = block(input_tensor=max_pooling)
    bath_norm = BatchNormalization()(block_output)
    avg_pool = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='same')(bath_norm)
    dropout = Dropout(0.2)(avg_pool)
    block_output_2 = block(input_tensor=dropout)
    bath_norm_2 = BatchNormalization()(block_output_2)
    avg_pool_2 = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='same')(bath_norm_2)
    dropout_2 = Dropout(0.2)(avg_pool_2)
    global_avg_pool = keras.layers.GlobalAveragePooling2D()(dropout_2)
    flatten_layer = Flatten()(global_avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model