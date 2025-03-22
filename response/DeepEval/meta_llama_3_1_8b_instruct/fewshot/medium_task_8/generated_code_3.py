import keras
import tensorflow as tf
from keras.layers import Input, Lambda, Conv2D, Add, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    def main_path(input_tensor):
        split = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        group1 = split[0]
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split[1])
        group2 = Lambda(lambda x: x + conv2)(split[1])
        conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(Add()([group1, group2]))
        output_tensor = Concatenate()([group1, conv3])
        return output_tensor

    main_path_output = main_path(input_layer)

    # Branch path
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Fuse main and branch paths
    adding_layer = Add()([main_path_output, conv1])

    # Final classification layer
    flatten_layer = Flatten()(adding_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model