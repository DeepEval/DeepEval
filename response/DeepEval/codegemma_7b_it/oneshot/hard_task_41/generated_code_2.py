import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, Dropout, Flatten, Dense, Reshape

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1: Parallel Pooling Paths
    def block_1(input_tensor):
        pool_1x1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        pool_2x2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        pool_4x4 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)

        pool_1x1_flatten = Flatten()(pool_1x1)
        pool_2x2_flatten = Flatten()(pool_2x2)
        pool_4x4_flatten = Flatten()(pool_4x4)

        concat = Concatenate()([pool_1x1_flatten, pool_2x2_flatten, pool_4x4_flatten])
        dropout = Dropout(0.5)(concat)

        return dropout

    block_1_output = block_1(input_tensor=input_layer)

    # Reshape and FC Layer
    dense_reshape = Dense(units=64, activation='relu')(block_1_output)
    reshape = Reshape((4, 4, 1))(dense_reshape)

    # Block 2: Multiple Branch Connections
    def block_2(input_tensor):
        branch_1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch_1x1_maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch_1x1)

        branch_3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch_3x3_maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch_3x3)

        branch_5x5 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch_5x5_maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch_5x5)

        branch_avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_tensor)
        branch_avg_pool_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_avg_pool)

        concat = Concatenate()([branch_1x1_maxpool, branch_3x3_maxpool, branch_5x5_maxpool, branch_avg_pool_conv])
        return concat

    block_2_output = block_2(input_tensor=reshape)

    # Classification Layers
    flatten = Flatten()(block_2_output)
    dense_1 = Dense(units=128, activation='relu')(flatten)
    dense_2 = Dense(units=10, activation='softmax')(dense_1)

    model = keras.Model(inputs=input_layer, outputs=dense_2)

    return model