import keras
from keras.layers import Input, Conv2D, Dropout, Add, Lambda, Concatenate, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Block 1
    # Main path
    main_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    main_path = Dropout(0.2)(main_path)
    main_path = Conv2D(filters=3, kernel_size=(3, 3), padding='same', activation='relu')(main_path)

    # Branch path
    branch_path = input_layer

    # Adding the outputs of main and branch paths
    block1_output = Add()([main_path, branch_path])

    # Block 2
    # Split the output into 3 groups using Lambda layer
    split_tensors = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(block1_output)

    # Feature extraction with separable convolutions
    sep_conv_outputs = []
    kernel_sizes = [1, 3, 5]
    for i, kernel_size in enumerate(kernel_sizes):
        sep_conv = Conv2D(filters=32, kernel_size=(kernel_size, kernel_size), padding='same', activation='relu')(split_tensors[i])
        sep_conv = Dropout(0.2)(sep_conv)
        sep_conv_outputs.append(sep_conv)

    # Concatenate the outputs from the three paths
    block2_output = Concatenate()(sep_conv_outputs)

    # Flattening and fully connected layer
    flatten_layer = Flatten()(block2_output)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)  # CIFAR-10 has 10 classes

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model