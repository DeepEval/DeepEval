import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, AveragePooling2D, Flatten, Dense

def basic_block(input_tensor, filters):
    conv = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
    bn = BatchNormalization()(conv)
    relu = Activation('relu')(bn)
    return relu

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    conv_init = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    bn_init = BatchNormalization()(conv_init)
    relu_init = Activation('relu')(bn_init)

    # First basic block
    block1_main_path = basic_block(relu_init, filters=16)
    block1_branch_path = basic_block(input_tensor=input_layer, filters=16)
    block1_output = Add()([block1_main_path, block1_branch_path])
    relu_block1 = Activation('relu')(block1_output)

    # Second basic block
    block2_main_path = basic_block(relu_block1, filters=32)
    block2_branch_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')(relu_block1)
    bn_branch = BatchNormalization()(block2_branch_path)
    block2_output = Add()([block2_main_path, bn_branch])
    relu_block2 = Activation('relu')(block2_output)

    # Average pooling layer
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(relu_block2)

    # Flattening and fully connected layer
    flatten = Flatten()(avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model