import keras
from keras.layers import Input, GlobalAveragePooling2D, Flatten, Dense, Reshape, Multiply, Add

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=input_tensor.shape[-1], activation='relu')(avg_pool)
        dense2 = Dense(units=input_tensor.shape[-1], activation='relu')(dense1)
        reshape = Reshape((1, 1, input_tensor.shape[-1]))(dense2)
        output_tensor = Multiply()([reshape, input_tensor])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)

    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)

    branch_output = block_1(input_tensor=max_pooling)

    main_path = Add()([block1_output, max_pooling])

    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv4)

    flatten = Flatten()(max_pooling2)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model