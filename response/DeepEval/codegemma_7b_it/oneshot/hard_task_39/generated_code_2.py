import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape, Conv2DTranspose, Dropout

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    max_pool1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_layer)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    max_pool3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(input_layer)
    flattened1 = Flatten()(max_pool1)
    flattened2 = Flatten()(max_pool2)
    flattened3 = Flatten()(max_pool3)
    concat_block1 = Concatenate()([flattened1, flattened2, flattened3])

    # Block 2
    reshape_layer = Reshape((1, 1, -1))(concat_block1)
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    conv3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    conv4 = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(reshape_layer)
    concat_block2 = Concatenate()([conv1, conv2, conv3, conv4])
    flatten_block2 = Flatten()(concat_block2)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(flatten_block2)

    # Create model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model