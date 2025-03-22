import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Lambda, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Main pathway
    conv_1x1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_branch_1x1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_branch_1x3 = Conv2D(filters=16, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_branch_3x1 = Conv2D(filters=16, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    concat_branch = Concatenate()([conv_branch_1x1, conv_branch_1x3, conv_branch_3x1])
    conv_1x1_main = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_branch)

    # Branch pathway
    conv_branch_1x1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_branch_3x3 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_branch_5x5 = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_branch_max_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)

    concat_branch = Concatenate()([conv_branch_1x1, conv_branch_3x3, conv_branch_5x5, conv_branch_max_pool])
    conv_1x1_branch = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_branch)

    # Fusion
    concat_main_branch = Concatenate()([conv_1x1_main, conv_1x1_branch])
    fused_output = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_main_branch)

    # Classification
    flatten = Flatten()(fused_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model