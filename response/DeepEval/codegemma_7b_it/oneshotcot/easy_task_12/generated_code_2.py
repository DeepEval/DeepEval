import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    # Branch path
    branch = Conv2D(filters=32, kernel_size=(1, 1), strides=(2, 2), padding='valid', activation='relu')(max_pooling)

    # Main path block 1
    conv_main_block1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling)
    conv_main_block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_main_block1)
    max_pooling_main_block = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_main_block2)

    # Branch path block 1
    conv_branch_block1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch)
    conv_branch_block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_branch_block1)
    max_pooling_branch_block = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_branch_block2)

    # Concatenate outputs from main and branch paths
    concat_layer = Concatenate()([max_pooling_main_block, max_pooling_branch_block])

    # Batch normalization and flattening
    batch_norm = BatchNormalization()(concat_layer)
    flatten_layer = Flatten()(batch_norm)

    # Fully connected layer
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Model construction
    model = keras.Model(inputs=input_layer, outputs=dense_layer)

    return model