import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Flatten, Add, Reshape

def dl_model():     
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    conv_main = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling_main = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_main)
    global_pooling_main = GlobalAveragePooling2D()(max_pooling_main)
    dense1_main = Dense(units=128, activation='relu')(global_pooling_main)
    dense2_main = Dense(units=32, activation='relu')(dense1_main)
    reshape_main = Reshape((32, 32, 32))(dense2_main)

    # Branch path
    conv_branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling_branch = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_branch)
    conv_branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling_branch)
    max_pooling_branch = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_branch)
    conv_branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling_branch)

    # Combine main and branch paths
    add_layers = Add()([reshape_main, conv_branch])

    # Fully connected layers
    flatten_layer = Flatten()(add_layers)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model