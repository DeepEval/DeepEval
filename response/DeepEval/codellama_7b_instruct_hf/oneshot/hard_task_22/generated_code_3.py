import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Main path
    input_layer = Input(shape=(32, 32, 3))
    split_input = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    group1 = split_input[0]
    group2 = split_input[1]
    group3 = split_input[2]
    group1_conv = Conv2D(32, (1, 1), activation='relu')(group1)
    group2_conv = Conv2D(32, (3, 3), activation='relu')(group2)
    group3_conv = Conv2D(32, (5, 5), activation='relu')(group3)
    main_path_output = Concatenate()([group1_conv, group2_conv, group3_conv])
    main_path_output = MaxPooling2D((2, 2))(main_path_output)
    main_path_output = BatchNormalization()(main_path_output)
    main_path_output = Flatten()(main_path_output)
    main_path_output = Dense(64, activation='relu')(main_path_output)
    main_path_output = Dense(10, activation='softmax')(main_path_output)

    # Branch path
    branch_path_input = Input(shape=(32, 32, 3))
    branch_path_output = Conv2D(32, (1, 1), activation='relu')(branch_path_input)
    branch_path_output = BatchNormalization()(branch_path_output)
    branch_path_output = Flatten()(branch_path_output)
    branch_path_output = Dense(64, activation='relu')(branch_path_output)
    branch_path_output = Dense(10, activation='softmax')(branch_path_output)

    # Fusion
    fusion_input = Concatenate()([main_path_output, branch_path_output])
    fusion_output = Dense(64, activation='relu')(fusion_input)
    fusion_output = Dense(10, activation='softmax')(fusion_output)

    # Create the model
    model = keras.Model(inputs=[input_layer, branch_path_input], outputs=fusion_output)

    return model