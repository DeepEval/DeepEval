import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    input_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)
    input_groups[0] = input_groups[0]
    input_groups[1] = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_groups[1])
    input_groups[2] = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_groups[2])
    main_path_output = Concatenate()([input_groups[0], input_groups[1], input_groups[2]])

    # Branch path
    branch_path_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Fuse main and branch paths
    fused_path_output = Add()([main_path_output, branch_path_output])

    # Flatten and classify
    flatten_layer = Flatten()(fused_path_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model