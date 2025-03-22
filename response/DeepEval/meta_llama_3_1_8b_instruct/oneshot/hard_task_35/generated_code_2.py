import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, Reshape, Multiply, Add

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)

    def same_block(input_tensor):
        global_avg_pool = GlobalAveragePooling2D()(input_tensor)
        weights = Dense(units=32, activation='relu')(global_avg_pool)
        weights = Reshape(target_shape=(1, 1, 32))(weights)
        element_wise_product = Multiply()([input_tensor, weights])
        return element_wise_product

    branch1 = same_block(max_pooling1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
    branch1 = same_block(max_pooling2)

    branch2 = same_block(max_pooling1)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    max_pooling3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)
    branch2 = same_block(max_pooling3)

    concat_output = Concatenate()([branch1, branch2])
    bath_norm = BatchNormalization()(concat_output)
    flatten_layer = Flatten()(bath_norm)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model