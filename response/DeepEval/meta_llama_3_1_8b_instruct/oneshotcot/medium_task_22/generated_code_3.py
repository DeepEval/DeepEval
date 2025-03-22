import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv_branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling_branch1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv_branch1)

    def branch2(input_tensor):
        conv_path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv_path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_path2)
        output_tensor = Concatenate()([conv_path1, conv_path2, path3])
        return output_tensor

    branch2_output = branch2(input_tensor=max_pooling_branch1)

    def branch3(input_tensor):
        path1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        output_tensor = path1
        return output_tensor

    branch3_output = branch3(input_tensor=max_pooling_branch1)

    multi_branch_concat = Concatenate()([max_pooling_branch1, branch2_output, branch3_output])
    batch_norm = BatchNormalization()(multi_branch_concat)
    flatten_layer = Flatten()(batch_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model