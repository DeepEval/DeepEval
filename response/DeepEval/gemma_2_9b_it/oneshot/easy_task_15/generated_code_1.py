import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Dropout, GlobalAveragePooling2D, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    
    def feature_block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
        pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)
        dropout = Dropout(rate=0.25)(pool)
        return dropout

    block1_output = feature_block(input_layer)
    block2_output = feature_block(block1_output)

    gap_layer = GlobalAveragePooling2D()(block2_output)
    flatten_layer = Flatten()(gap_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model