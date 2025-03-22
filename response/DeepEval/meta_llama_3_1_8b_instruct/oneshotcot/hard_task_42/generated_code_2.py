import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Reshape

def dl_model():
    
    # Define block 1
    def block1(input_tensor):
        path1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        path2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        path3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        output_tensor1 = Concatenate()([path1, path2, path3])
        output_tensor1 = Flatten()(output_tensor1)
        output_tensor1 = Dropout(0.2)(output_tensor1)
        return output_tensor1

    # Define block 2
    def block2(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path2)
        path2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path2)
        path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3)
        path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(path4)
        output_tensor2 = Concatenate()([path1, path2, path3, path4])
        output_tensor2 = Flatten()(output_tensor2)
        output_tensor2 = Dropout(0.2)(output_tensor2)
        return output_tensor2

    input_layer = Input(shape=(28, 28, 1))
    
    block_output1 = block1(input_layer)
    reshape_layer = Reshape((block_output1.shape[-1], 1))(block_output1)
    dense_layer = Dense(units=128, activation='relu')(reshape_layer)
    output_block1 = block_output1
    
    output_block2 = block2(input_layer)
    concat_layer = Concatenate()([output_block1, output_block2])
    flatten_layer = Flatten()(concat_layer)
    dense_layer1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model