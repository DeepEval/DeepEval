import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, AveragePooling2D, GlobalAveragePooling2D, Dense, Reshape

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # First block
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv1_2)

    # Main path
    conv2_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling1)
    conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_1)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv2_2)

    # Branch path
    avg_pooling = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='valid')(conv1_2)
    flatten_avg = Flatten()(avg_pooling)

    # Second block
    # Main path
    conv3_1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling2)
    conv3_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3_1)
    conv3_3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3_2)
    global_avg_pool = GlobalAveragePooling2D()(conv3_3)

    # Branch path
    flatten_input = Flatten()(input_layer)
    dense1 = Dense(units=64, activation='relu')(flatten_input)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Combine outputs
    concat = Add()([global_avg_pool, dense2])
    reshape = Reshape(target_shape=(1, 1, 64))(concat)

    # Classification layer
    flatten = Flatten()(reshape)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model