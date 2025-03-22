import keras
from keras.layers import Input, Conv2D, Flatten, Dense, Lambda, Add, Concatenate
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Split the input into three groups along the last dimension
    split_input = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Main path
    group1 = split_input[0]  # Unchanged
    group2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_input[1])  # 3x3 conv
    group3 = split_input[2]  # Unchanged

    # Combine group2 and group3
    combined_group = Add()([group2, group3])
    main_path_output = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(combined_group)

    # Branch path
    branch_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Fuse the outputs from the main and branch paths
    fused_output = Add()([main_path_output, branch_output])

    # Flatten and fully connected layer for classification
    flatten_layer = Flatten()(fused_output)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)  # 10 classes for CIFAR-10

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model