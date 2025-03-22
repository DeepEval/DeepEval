import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense, Reshape

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    avg_pool_1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    avg_pool_2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    avg_pool_4 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)
    
    flat_1 = Flatten()(avg_pool_1)
    flat_2 = Flatten()(avg_pool_2)
    flat_3 = Flatten()(avg_pool_4)
    
    concat_block1 = Concatenate()([flat_1, flat_2, flat_3])
    
    dense_block1 = Dense(units=128, activation='relu')(concat_block1)
    reshape_layer = Reshape((128, 1))(dense_block1)  

    # Block 2
    conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    conv1_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(reshape_layer)

    conv2_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    conv2_7 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    conv2_7_1 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    conv2_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    avg_pool2 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(reshape_layer)

    concat_block2 = Concatenate()([conv1_1, conv1_3, avg_pool1, conv2_1, conv2_7, conv2_7_1, conv2_3, avg_pool2])

    # Final Layers
    dense3 = Dense(units=64, activation='relu')(concat_block2)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model