import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Dropout, Flatten, Dense, GlobalAveragePooling2D

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    
    def feature_block(input_tensor):
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        x = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
        x = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
        x = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
        x = Dropout(0.25)(x)
        return x

    block1_output = feature_block(input_layer)
    block2_output = feature_block(block1_output)

    global_avg_pool = GlobalAveragePooling2D()(block2_output)
    flatten_layer = Flatten()(global_avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model