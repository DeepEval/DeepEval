import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    def block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
        avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)
        drop_out = Dropout(0.2)(avg_pool)
        return drop_out

    block1_output = block(input_layer)
    block2_output = block(block1_output)
    
    global_avg_pool = GlobalAveragePooling2D()(block2_output)
    flatten_layer = Flatten()(global_avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model