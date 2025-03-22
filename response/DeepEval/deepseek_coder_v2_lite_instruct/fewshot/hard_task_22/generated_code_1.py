import keras
from keras.layers import Input, Conv2D, SeparableConv2D, MaxPooling2D, Add, Flatten, Dense, Lambda
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path_input = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_layer)
    conv1_1 = SeparableConv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path_input[0])
    conv1_3 = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path_input[1])
    conv1_5 = SeparableConv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(main_path_input[2])
    main_path_output = Add()([conv1_1, conv1_3, conv1_5])
    main_path_output = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(main_path_output)

    # Branch path
    branch_path_input = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_path_output = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(branch_path_input)

    # Fusion of main path and branch path
    fused_output = Add()([main_path_output, branch_path_output])

    # Flatten and fully connected layers
    flatten_layer = Flatten()(fused_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model