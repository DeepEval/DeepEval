import keras
from keras.layers import Input, Conv2D, BatchNormalization, Add, Flatten, Dense, AveragePooling2D

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Basic block for the first level
    def basic_block(input_tensor):
        conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        bn1 = BatchNormalization()(conv1)
        conv2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(bn1)
        bn2 = BatchNormalization()(conv2)
        output_tensor = Add()([input_tensor, bn2])

        return output_tensor

    level1 = basic_block(conv1)

    # Basic block for the second level
    def residual_block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        bn1 = BatchNormalization()(conv1)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(bn1)
        bn2 = BatchNormalization()(conv2)
        branch_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        bn_branch = BatchNormalization()(branch_conv)
        output_tensor = Add()([bn_branch, bn2])

        return output_tensor

    level2 = residual_block(level1)
    level2 = residual_block(level2)

    # Global branch for the third level
    global_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Third level
    level3 = Add()([level2, global_conv])

    avg_pool = AveragePooling2D(pool_size=(4, 4), strides=4, padding='valid')(level3)
    flatten = Flatten()(avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model