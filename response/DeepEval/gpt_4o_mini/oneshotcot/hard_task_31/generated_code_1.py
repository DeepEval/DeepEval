import keras
from keras.layers import Input, Conv2D, Dropout, Add, Lambda, Concatenate, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block
    # Main path
    main_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    main_path = Dropout(0.5)(main_path)
    main_path = Conv2D(filters=3, kernel_size=(3, 3), padding='same', activation='relu')(main_path)

    # Branch path
    branch_path = input_layer

    # Combine both paths
    block1_output = Add()([main_path, branch_path])

    # Second block
    # Splitting the output into three groups along the last dimension
    split_tensors = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(block1_output)

    # Creating separable convolutions for each split
    sep_conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_tensors[0])
    sep_conv1 = Dropout(0.5)(sep_conv1)

    sep_conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_tensors[1])
    sep_conv2 = Dropout(0.5)(sep_conv2)

    sep_conv3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_tensors[2])
    sep_conv3 = Dropout(0.5)(sep_conv3)

    # Concatenate the outputs of the three separable convolutions
    block2_output = Concatenate()([sep_conv1, sep_conv2, sep_conv3])

    # Flatten and fully connected layer
    flatten_layer = Flatten()(block2_output)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model