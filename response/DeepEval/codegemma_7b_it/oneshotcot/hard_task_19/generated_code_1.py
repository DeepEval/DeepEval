import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, Add, Reshape

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    main_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(main_conv1)
    main_conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(main_conv2)
    main_max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(main_conv3)

    # Branch path
    branch_avg_pool = GlobalAveragePooling2D()(input_layer)
    branch_dense1 = Dense(units=128, activation='relu')(branch_avg_pool)
    branch_dense2 = Dense(units=128, activation='relu')(branch_dense1)
    branch_dense2_reshaped = Reshape((1, 1, 128))(branch_dense2)
    branch_up_sampled = keras.layers.UpSampling2D(size=(8, 8))(branch_dense2_reshaped)

    # Combine paths
    combined = Add()([main_max_pool, branch_up_sampled])

    # Classification layers
    flatten_layer = Flatten()(combined)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    dense3 = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=dense3)

    return model