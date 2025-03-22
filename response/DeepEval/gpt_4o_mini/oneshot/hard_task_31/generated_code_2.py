import keras
from keras.layers import Input, Conv2D, Dropout, Add, Lambda, SeparableConv2D, Flatten, Dense, Concatenate
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block
    # Main path: Convolution -> Dropout -> Convolution
    main_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    main_path = Dropout(0.5)(main_path)
    main_path = Conv2D(filters=3, kernel_size=(3, 3), padding='same', activation='relu')(main_path)

    # Branch path: Direct connection to input
    branch_path = input_layer

    # Adding both paths together
    block_output = Add()([main_path, branch_path])

    # Second block: Splitting and using separable convolutions
    split_tensors = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(block_output)

    # Applying separable convolutions with dropout to each split
    group1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_tensors[0])
    group1 = Dropout(0.5)(group1)

    group2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_tensors[1])
    group2 = Dropout(0.5)(group2)

    group3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_tensors[2])
    group3 = Dropout(0.5)(group3)

    # Concatenating outputs from the three groups
    concatenated_output = Concatenate()([group1, group2, group3])

    # Final layers: Flatten and fully connected layer
    flatten_layer = Flatten()(concatenated_output)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model