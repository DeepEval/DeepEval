import keras
from keras.layers import Input, Conv2D, Lambda, Add, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    main_split = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    main_path1 = main_split[0]
    main_path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_split[1])
    main_path3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_split[2])
    main_path3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path3)
    main_path_output = Add()([main_path1, main_path2, main_path3])

    # Branch Path
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Fuse Main and Branch Paths
    fused_output = Add()([main_path_output, branch_path])

    # Flatten and Fully Connected Layer
    flatten_layer = Flatten()(fused_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model