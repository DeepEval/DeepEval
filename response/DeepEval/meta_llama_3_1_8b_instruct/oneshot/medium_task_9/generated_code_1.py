import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    def basic_block(input_tensor):
        conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        batch_norm = BatchNormalization()(conv)
        relu = ReLU()(batch_norm)
        return relu

    branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    output = Add()([basic_block(input_tensor=conv1), branch])

    block1 = basic_block(input_tensor=output)
    block2 = basic_block(input_tensor=block1)

    branch2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2)
    output = Add()([block2, branch2])

    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(output)
    flatten_layer = Flatten()(avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model