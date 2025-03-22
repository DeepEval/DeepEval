import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block1_output = Add()([main_path, branch_path])

    # Block 2
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(block1_output)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(block1_output)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(block1_output)
    block2_output = Concatenate()([pool1, pool2, pool3])
    
    flatten_layer = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model