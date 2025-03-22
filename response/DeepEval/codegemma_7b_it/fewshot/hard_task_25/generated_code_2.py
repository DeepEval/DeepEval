import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Flatten, Concatenate, DepthwiseConv2D, Dense, Reshape, Conv2DTranspose

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Main path
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path_branch_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1)
    main_path_branch_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1_1)
    main_path_branch_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path_branch_2)
    main_path_branch_3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(conv1_1)
    main_path_branch_3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path_branch_3)
    main_path_branch_3 = Conv2DTranspose(filters=32, kernel_size=(4, 4), strides=(4, 4), padding='same', activation='relu')(main_path_branch_3)
    main_path_concat = Concatenate()([main_path_branch_1, main_path_branch_2, main_path_branch_3])
    main_path_output = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path_concat)

    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_path_output = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_path)

    # Fuse main path and branch path outputs
    fused_output = Add()([main_path_output, branch_path_output])

    # Final output
    flatten_layer = Flatten()(fused_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model