import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, GlobalAveragePooling2D

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)

    def block(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3, path4])
        return output_tensor

    block_output = block(input_tensor=max_pooling)

    # Additional block for local feature capture
    block_output = block(input_tensor=block_output)

    avg_pooling = GlobalAveragePooling2D()(block_output)
    flatten_layer = Flatten()(avg_pooling)
    dropout = Dropout(rate=0.2)(flatten_layer)
    dense = Dense(units=128, activation='relu')(dropout)
    output_layer = Dense(units=10, activation='softmax')(dense)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model