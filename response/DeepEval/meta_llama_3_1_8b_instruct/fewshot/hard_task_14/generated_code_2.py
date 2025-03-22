import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Conv2D, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 dataset

    main_path = GlobalAveragePooling2D()(input_layer)
    dense1 = Dense(units=64, activation='relu')(main_path)
    dense2 = Dense(units=64, activation='relu')(dense1)
    weights = Dense(units=3, activation='linear')(dense2)
    reshaped_weights = Reshape(target_shape=(3, 1, 1))(weights)
    weighted_feature_map = Multiply()([input_layer, reshaped_weights])

    branch_path = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

    adding_layer = Add()([weighted_feature_map, branch_path])

    flatten_layer = Flatten()(adding_layer)
    dense3 = Dense(units=64, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model