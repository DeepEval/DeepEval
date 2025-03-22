import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Lambda, Concatenate, Dense, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path_group1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path_group2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path_group3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path_output = Concatenate()([main_path_group1, main_path_group2, main_path_group3])

    # Branch path
    branch_path_output = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Fusion of main and branch paths
    fused_features = Add()([main_path_output, branch_path_output])

    # Flatten and classification
    flatten_layer = Flatten()(fused_features)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model