import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, Reshape, Multiply

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)

    def block(input_tensor):

        pool = GlobalAveragePooling2D()(input_tensor)
        flat = Flatten()(pool)
        weights = Dense(units=3, activation='linear')(flat)
        weights = Reshape(target_shape=(3, 1))(weights)
        weighted_input = Multiply()([input_tensor, weights])
        output_tensor = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(weighted_input)
        output_tensor = BatchNormalization()(output_tensor)
        return output_tensor
    
    block_output1 = block(input_tensor=max_pooling1)
    block_output2 = block(input_tensor=max_pooling1)
    concat = Concatenate()([block_output1, block_output2])
    flatten_layer = Flatten()(concat)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model