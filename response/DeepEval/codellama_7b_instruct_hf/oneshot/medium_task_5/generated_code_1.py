import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    # Main path
    main_input = Input(shape=(32, 32, 3))
    main_conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_input)
    main_conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_conv1)
    main_maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(main_conv2)
    main_block1 = BatchNormalization()(main_maxpool)
    main_block2 = BatchNormalization()(main_block1)
    main_output = Concatenate()([main_block1, main_block2])

    # Branch path
    branch_input = Input(shape=(32, 32, 3))
    branch_conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_input)
    branch_maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(branch_conv1)
    branch_output = BatchNormalization()(branch_maxpool)

    # Combine main and branch paths
    combined_output = Concatenate()([main_output, branch_output])

    # Flatten and project onto a probability distribution
    flatten = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=[main_input, branch_input], outputs=dense2)

    return model