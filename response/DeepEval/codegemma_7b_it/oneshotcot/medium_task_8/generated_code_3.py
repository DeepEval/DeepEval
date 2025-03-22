import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = input_layer
    main_path_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(main_path)
    main_path_group1 = main_path_groups[0]
    main_path_group2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path_groups[1])
    main_path_group3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path_groups[2])
    main_path_output = Concatenate()([main_path_group1, main_path_group2, main_path_group3])

    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Fusion
    fusion = Add()([main_path_output, branch_path])

    # Classification
    flatten_layer = Flatten()(fusion)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Model definition
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model