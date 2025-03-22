import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Lambda, Concatenate
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path_group1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    main_path_group2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    main_path_group3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(input_layer)
    main_path_output = Concatenate()([main_path_group1, main_path_group2, main_path_group3])

    # Branch path
    branch_path_output = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Fusion of main and branch paths
    fused_features = Add()([main_path_output, branch_path_output])

    # Flatten and fully connected layers
    flattened_features = Flatten()(fused_features)
    dense1 = Dense(units=128, activation='relu')(flattened_features)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model