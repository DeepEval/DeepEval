import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    conv_init = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 1: Local features extraction
    conv_branch_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_init)

    # Branch 2: Downsampling, max pooling, and upsampling
    max_pool_branch_2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv_init)
    conv_branch_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool_branch_2)
    up_sample_branch_2 = UpSampling2D(size=(2, 2))(conv_branch_2)

    # Branch 3: Downsampling, max pooling, and upsampling
    max_pool_branch_3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv_init)
    conv_branch_3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool_branch_3)
    up_sample_branch_3 = UpSampling2D(size=(4, 4))(conv_branch_3)

    # Concatenate outputs of all branches
    concat_branches = Concatenate()([conv_branch_1, up_sample_branch_2, up_sample_branch_3])

    # Fusing branches with 1x1 convolutional layer
    conv_fused = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_branches)

    # Batch normalization
    batch_norm = BatchNormalization()(conv_fused)

    # Flattening and fully connected layers
    flatten_layer = Flatten()(batch_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model creation
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model