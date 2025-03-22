import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, GlobalAveragePooling2D

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)

    def block(input_tensor):

        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
        avg_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_tensor)
        output_tensor = Concatenate()([conv1, conv2, conv3, avg_pool])

        return output_tensor
        
    block_output = block(input_tensor=max_pooling)
    dropout = Dropout(rate=0.25)(block_output)
    block_output_2 = block(input_tensor=dropout)
    avg_pool_2 = GlobalAveragePooling2D()(block_output_2)
    flatten_layer = Flatten()(avg_pool_2)
    dense1 = Dense(units=64, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model