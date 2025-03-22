import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    split_input = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_layer)

    # Main Path
    main_path = []
    for i in range(3):
        conv = Conv2D(filters=32, kernel_size=(i+1, i+1), strides=(1, 1), padding='same', activation='relu')(split_input[i])
        max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)
        main_path.append(max_pooling)

    main_path_output = Concatenate()(main_path)

    # Branch Path
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_path_output = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch_path)

    # Fusion
    combined_output = keras.layers.add([main_path_output, branch_path_output])

    # Classification
    flatten_layer = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model