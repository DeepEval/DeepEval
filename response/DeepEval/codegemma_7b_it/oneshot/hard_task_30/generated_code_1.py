import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Reshape

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # First block: Dual-path structure
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same')(main_path)

    branch_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_path = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same')(branch_path)

    combined_output = keras.layers.Add()([main_path, branch_path])

    # Second block: Grouped depthwise separable convolutional layers
    grouped_input = Lambda(lambda x: tf.split(x, 3, axis=3))(combined_output)
    group1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(grouped_input[0])
    group1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', use_bias=False)(group1)
    group1 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same')(group1)

    group2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(grouped_input[1])
    group2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', use_bias=False)(group2)
    group2 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same')(group2)

    group3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(grouped_input[2])
    group3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', use_bias=False)(group3)
    group3 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same')(group3)

    concat_groups = Concatenate()([group1, group2, group3])

    # Output layers
    flatten_layer = Flatten()(concat_groups)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model