import keras
from keras.layers import Input, Conv2D, BatchNormalization, Add, ReLU, AveragePooling2D, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    def basic_block(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(input_tensor)
        batch_norm = BatchNormalization()(conv)
        relu = ReLU()(batch_norm)
        branch = input_tensor
        adding_layer = Add()([relu, branch])
        return adding_layer
    
    block1_output = basic_block(input_layer)
    
    block2_output = basic_block(block1_output)
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(block2_output)
    batch_norm = BatchNormalization()(conv1)
    relu = ReLU()(batch_norm)
    branch = block2_output
    adding_layer = Add()([relu, branch])
    block2_output = adding_layer
    
    block3_output = basic_block(block2_output)
    conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(block3_output)
    batch_norm = BatchNormalization()(conv2)
    relu = ReLU()(batch_norm)
    branch = block3_output
    adding_layer = Add()([relu, branch])
    block3_output = adding_layer
    
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(input_layer)
    batch_norm = BatchNormalization()(conv3)
    relu = ReLU()(batch_norm)
    adding_layer = Add()([relu, block3_output])
    global_branch_output = adding_layer
    
    global_branch_output = AveragePooling2D(pool_size=(8, 8), strides=1, padding='same')(global_branch_output)
    flatten = Flatten()(global_branch_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model