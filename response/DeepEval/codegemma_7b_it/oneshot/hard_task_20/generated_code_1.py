import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)
    main_path_1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path[0])
    main_path_3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path[1])
    main_path_5x5 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(main_path[2])

    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Fusion
    fused_features = Concatenate()([main_path_1x1, main_path_3x3, main_path_5x5, branch_path])

    # Classification
    flatten_layer = Flatten()(fused_features)
    dense1 = Dense(units=64, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model