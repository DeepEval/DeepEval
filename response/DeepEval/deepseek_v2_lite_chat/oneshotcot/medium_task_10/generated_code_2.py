import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Add, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    bn1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(bn1)

    def block(input_tensor):
        def residual_block(input_tensor):
            conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
            bn = BatchNormalization()(conv)
            return Add()([bn, input_tensor])

        conv2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
        bn2 = BatchNormalization()(conv2)
        branch = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        residual = Add()([bn2, branch])
        return residual

    block_output = block(input_tensor=pool1)
    block_output = residual_block(input_tensor=block_output)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(block_output)

    global_branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
    output = Add()([pool2, global_branch])
    output = Flatten()(output)
    dense1 = Dense(units=128, activation='relu')(output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model