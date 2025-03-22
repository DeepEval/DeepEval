import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1: Main path and Branch path
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    branch_path = input_layer
    output_block1 = Add()([main_path, branch_path])

    # Block 2: Multi-scale Max Pooling
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(output_block1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(output_block1)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(output_block1)
    output_block2 = Concatenate()([pool1, pool2, pool3])

    flatten_layer = Flatten()(output_block2)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model