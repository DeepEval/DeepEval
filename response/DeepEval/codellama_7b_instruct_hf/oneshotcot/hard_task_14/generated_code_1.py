import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Main path
    input_layer = Input(shape=(32, 32, 3))
    global_average_pooling = GlobalAveragePooling2D()(input_layer)
    flatten = Flatten()(global_average_pooling)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    branch_path_output = Dense(units=3, activation='softmax')(dense2)

    # Branch path
    branch_path_input = Input(shape=(32, 32, 3))
    branch_path_conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_path_input)
    branch_path_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(branch_path_conv)
    branch_path_dense1 = Dense(units=128, activation='relu')(branch_path_pooling)
    branch_path_dense2 = Dense(units=64, activation='relu')(branch_path_dense1)
    branch_path_output = Dense(units=3, activation='softmax')(branch_path_dense2)

    # Combine main and branch paths
    concatenated_output = Concatenate()([main_path_output, branch_path_output])

    # Final classification
    final_dense = Dense(units=128, activation='relu')(concatenated_output)
    final_output = Dense(units=10, activation='softmax')(final_dense)

    model = keras.Model(inputs=[input_layer, branch_path_input], outputs=final_output)
    return model