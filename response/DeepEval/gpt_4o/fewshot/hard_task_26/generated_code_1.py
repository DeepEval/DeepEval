import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 1
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_conv1)

    # Branch 2
    branch2_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(main_conv1)
    branch2_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2_pool)
    branch2_upsample = UpSampling2D(size=(2, 2))(branch2_conv)

    # Branch 3
    branch3_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(main_conv1)
    branch3_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3_pool)
    branch3_upsample = UpSampling2D(size=(2, 2))(branch3_conv)

    # Concatenating branches
    concatenated_branches = Concatenate()([branch1, branch2_upsample, branch3_upsample])
    final_main_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated_branches)

    # Branch path
    branch_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Adding main path and branch path
    merged_paths = Add()([final_main_path, branch_conv1])

    # Fully connected layers
    flatten_layer = Flatten()(merged_paths)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Creating the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model