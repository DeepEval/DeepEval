import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Add, Lambda, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    def main_path(input_tensor):
        # Split the input into three groups
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)

        # Convolutions with different kernel sizes on each group
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])

        # Concatenate the outputs of the convolutions
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    # Branch path
    def branch_path(input_tensor):
        # 1x1 convolution to align the number of channels
        output_tensor = Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return output_tensor

    main_path_output = main_path(input_layer)
    branch_path_output = branch_path(input_layer)

    # Combine outputs of the main and branch paths
    fused_features = Add()([main_path_output, branch_path_output])

    # Flatten and fully connected layers for classification
    flatten_layer = Flatten()(fused_features)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model