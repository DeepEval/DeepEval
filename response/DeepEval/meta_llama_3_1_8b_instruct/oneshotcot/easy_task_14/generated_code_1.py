import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Reshape, Multiply, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv)

    global_avg_pool = GlobalAveragePooling2D()(max_pooling)

    def correlation_block(input_tensor):

        weights1 = Dense(units=32, activation='relu')(global_avg_pool)
        weights1 = Reshape((1, 1, 32))(weights1)

        weights2 = Dense(units=32, activation='relu')(global_avg_pool)
        weights2 = Reshape((1, 1, 32))(weights2)

        output_tensor = Multiply()([input_tensor, weights1])
        output_tensor = Multiply()([output_tensor, weights2])

        return output_tensor
        
    correlation_output = correlation_block(max_pooling)
    flatten_layer = Flatten()(correlation_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model