import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add
from keras.layers import GlobalAveragePooling2D, Dropout

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # Main pathway
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(conv)
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(conv1x1)
    pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1x1)
    dropout = Dropout(0.5)(pool)
    block_output = pool + conv1x1

    # Branch pathway
    conv_branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool)

    # Fuse the outputs from both pathways
    fuse = Add()([block_output, conv_branch])

    # Global average pooling and flatten
    gap = GlobalAveragePooling2D()(fuse)
    flatten_layer = Flatten()(gap)

    # Dense layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model