import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Dense, Reshape, Multiply

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # First block with four parallel branches
    conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    output_tensor = Concatenate()([conv1_1, conv1_2, conv1_3, conv1_4])

    # Second block with global average pooling and fully connected layers
    block2_output = GlobalAveragePooling2D()(output_tensor)
    dense1 = Dense(units=128, activation='relu')(block2_output)
    dense2 = Dense(units=128, activation='relu')(dense1)
    weights = Dense(units=128, activation='linear')(block2_output)
    weights = Reshape((128,))(weights)
    element_wise_product = Multiply()([dense2, weights])
    output_layer = Dense(units=10, activation='softmax')(element_wise_product)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model