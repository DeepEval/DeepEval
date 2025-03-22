import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Basic block
    block = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    batch_norm = BatchNormalization()(block)
    relu = ReLU()(batch_norm)

    # Feature fusion
    concat = Concatenate()([block, relu])

    # Main structure of the model
    main_block = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat)
    main_block_batch_norm = BatchNormalization()(main_block)
    main_block_relu = ReLU()(main_block_batch_norm)

    # Branch for feature extraction
    branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat)
    branch_batch_norm = BatchNormalization()(branch)
    branch_relu = ReLU()(branch_batch_norm)

    # Combine features
    combine = Add()([main_block_relu, branch_relu])

    # Average pooling layer
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(combine)

    # Flatten
    flat = Flatten()(avg_pool)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output)

    return model