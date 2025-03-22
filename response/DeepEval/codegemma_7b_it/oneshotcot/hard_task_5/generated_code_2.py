import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Concatenate, BatchNormalization, Flatten, Dense, Reshape, Permute, Add

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    split_input = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)
    path1 = Conv2D(filters=32 // 3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_input[0])
    path2 = Conv2D(filters=32 // 3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_input[1])
    path3 = Conv2D(filters=32 // 3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_input[2])
    concat_path1_path2_path3 = Concatenate(axis=3)([path1, path2, path3])

    # Block 2
    input_shape = keras.backend.int_shape(concat_path1_path2_path3)
    reshape_input_tensor = Reshape((input_shape[1], input_shape[2], 3, input_shape[3] // 3))(concat_path1_path2_path3)
    permute_input_tensor = Permute((1, 2, 3, 0))(reshape_input_tensor)
    reshape_input_tensor_2 = Reshape((input_shape[1], input_shape[2], input_shape[3] // 3, 3))(permute_input_tensor)

    # Block 3
    path4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu')(reshape_input_tensor_2)

    # Branch
    branch_input = Input(shape=(32, 32, 3))
    branch_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu')(branch_input)

    # Concatenation and Fully Connected Layer
    concat_branch_path4 = Add()([branch_conv, path4])
    flatten_layer = Flatten()(concat_branch_path4)
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=[input_layer, branch_input], outputs=dense_layer)

    return model