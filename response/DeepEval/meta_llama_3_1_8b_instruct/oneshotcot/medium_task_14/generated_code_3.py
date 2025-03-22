import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3)) # CIFAR-10 dataset has images of size 32x32x3

    def block(input_tensor):

        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        batch_norm = BatchNormalization()(conv)
        activation = ReLU()(batch_norm)

        return activation
    
    block1_output = block(input_layer)
    block2_output = block(block1_output)
    block3_output = block(block2_output)

    parallel_branch = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    parallel_branch = BatchNormalization()(parallel_branch)
    parallel_branch = ReLU()(parallel_branch)

    output_tensor = Add()([block1_output, block2_output, block3_output, parallel_branch])

    avg_pool = GlobalAveragePooling2D()(output_tensor)
    dense1 = Dense(units=64, activation='relu')(avg_pool)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model