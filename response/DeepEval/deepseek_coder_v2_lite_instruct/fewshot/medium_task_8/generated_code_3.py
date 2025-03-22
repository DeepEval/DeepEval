import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Add, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_split = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    main_path_1 = main_split[0]  # Remains unchanged
    main_path_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_split[1])
    main_path_3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(Add()([main_path_2, main_split[2]]))
    main_path_output = main_path_3

    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Fusion of main and branch paths
    fusion_output = Add()([main_path_output, branch_path])

    # Flatten and fully connected layer
    flatten_layer = Flatten()(fusion_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model