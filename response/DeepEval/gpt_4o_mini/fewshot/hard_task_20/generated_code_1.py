import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Concatenate, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32x3

    # Main path
    def main_path(input_tensor):
        split_tensors = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)

        # Feature extraction with different kernel sizes
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensors[0])
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_tensors[1])
        conv3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_tensors[2])

        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    # Branch path
    def branch_path(input_tensor):
        branch_conv = Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return branch_conv

    main_path_output = main_path(input_layer)
    branch_path_output = branch_path(input_layer)

    # Combine main and branch paths
    fused_features = Add()([main_path_output, branch_path_output])

    # Classification layers
    flatten_layer = Flatten()(fused_features)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model