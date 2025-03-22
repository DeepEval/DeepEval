import keras
from keras.layers import Input, Conv2D, Flatten, Dense, Lambda, Add, Concatenate
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Main path
    split_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    first_group = split_groups[0]
    second_group = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(split_groups[1])
    third_group = split_groups[2]

    # Combine second and third groups
    combined = Add()([second_group, third_group])  # Element-wise addition
    main_path_output = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(combined)

    # Concatenate outputs of all three groups
    concatenated = Concatenate()([first_group, main_path_output, third_group])

    # Branch path
    branch_path_output = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Fusing main and branch paths
    fused_output = Add()([concatenated, branch_path_output])

    # Flatten and fully connected layer
    flatten_layer = Flatten()(fused_output)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)  # 10 classes for CIFAR-10

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model