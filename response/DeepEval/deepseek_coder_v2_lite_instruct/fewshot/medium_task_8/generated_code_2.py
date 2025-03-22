import keras
from keras.layers import Input, Conv2D, Lambda, Add, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Main Path
    split_layer = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    main_path_group1 = split_layer[0]
    main_path_group2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_layer[1])
    main_path_group3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_layer[2])
    combined_main_path = Add()([main_path_group2, main_path_group3])
    main_path_output = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(combined_main_path)

    # Branch Path
    branch_path_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Fusion
    fused_output = Add()([main_path_output, branch_path_output])

    # Flatten and Fully Connected Layer
    flattened_output = Flatten()(fused_output)
    output_layer = Dense(units=10, activation='softmax')(flattened_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model