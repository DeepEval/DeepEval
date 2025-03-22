import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, Dense, GlobalAveragePooling2D, Reshape, Multiply

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
    gap = GlobalAveragePooling2D()(pool1)
    dense1 = Dense(units=32, activation='relu')(gap)
    weights = Dense(units=3, activation='linear')(dense1)
    reshaped_weights = Reshape(target_shape=(1, 1, 3))(weights)
    weighted_feature_map = Multiply()([pool1, reshaped_weights])
    main_path = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(weighted_feature_map)

    # Branch Path
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine Main and Branch Paths
    adding_layer = Add()([main_path, branch_path])

    # Classification Layers
    flatten_layer = Flatten()(adding_layer)
    dense2 = Dense(units=64, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model