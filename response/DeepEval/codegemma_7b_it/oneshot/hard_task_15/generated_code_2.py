import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, Multiply

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)

    def main_path(input_tensor):
        gap = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=input_tensor.shape[3], activation='relu')(gap)
        dense2 = Dense(units=input_tensor.shape[3], activation='sigmoid')(dense1)
        dense2 = keras.backend.reshape(dense2, (-1, 1, 1, input_tensor.shape[3]))
        output_tensor = Multiply()([input_tensor, dense2])

        return output_tensor
    
    main_output = main_path(input_tensor=max_pooling)

    def branch_path(input_tensor):
        return input_tensor
    
    branch_output = branch_path(input_tensor=input_layer)

    combined_output = Concatenate()([main_output, branch_output])
    bath_norm = BatchNormalization()(combined_output)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model