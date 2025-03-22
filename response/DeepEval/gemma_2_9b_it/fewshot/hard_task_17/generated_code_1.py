import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Conv2D, MaxPooling2D, Add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    x = GlobalAveragePooling2D()(input_layer)
    x = Dense(units=128, activation='relu')(x)
    x = Dense(units=64, activation='relu')(x)
    x = Reshape(target_shape=(32, 32, 64))(x)
    weighted_features = input_layer * x

    # Block 2
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(weighted_features)
    conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)

    # Fusion
    branch_output = pool 
    main_path_output = branch_output 
    combined_output = Add()([main_path_output, branch_output])

    # Classification
    flatten = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model