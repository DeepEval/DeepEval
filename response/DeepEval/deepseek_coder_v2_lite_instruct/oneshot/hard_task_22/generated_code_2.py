import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, MaxPooling2D, Add, Lambda, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    main_path = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_layer)
    main_path_1x1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(main_path[0])
    main_path_3x3 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(main_path[1])
    main_path_5x5 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(main_path[2])
    main_path_output = Concatenate(axis=-1)([main_path_1x1, main_path_3x3, main_path_5x5])

    # Branch Path
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Fusion
    fused_output = Add()([main_path_output, branch_path])

    # Flattening and Fully Connected Layers
    flatten_layer = Flatten()(fused_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model