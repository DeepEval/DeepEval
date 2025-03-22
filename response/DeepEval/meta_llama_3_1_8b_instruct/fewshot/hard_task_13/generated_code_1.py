import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Dense, Reshape, Multiply

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block with four parallel branches
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

    concat_output = Concatenate()([branch1, branch2, branch3, branch4])
    
    # Second block with global average pooling and fully connected layers
    global_pool = GlobalAveragePooling2D()(concat_output)
    dense1 = Dense(units=128, activation='relu')(global_pool)
    dense2 = Dense(units=64, activation='relu')(dense1)
    weights = Dense(units=32)(dense2)
    weights = Reshape(target_shape=(1, 1, 32))(weights)
    multiplied = Multiply()([concat_output, weights])
    output_layer = Dense(units=10, activation='softmax')(multiplied)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model