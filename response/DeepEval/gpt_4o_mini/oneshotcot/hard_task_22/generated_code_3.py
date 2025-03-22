import keras
from keras.layers import Input, Lambda, SeparableConv2D, Concatenate, Flatten, Dense, Add
import tensorflow as tf

def dl_model():
    # Step 1: Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Step 2: Split input into three groups along the channel dimension
    split_inputs = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Step 3: Main path with multi-scale feature extraction
    path1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_inputs[0])
    path2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_inputs[1])
    path3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_inputs[2])

    # Step 4: Concatenate outputs from the main path
    main_path_output = Concatenate()([path1, path2, path3])

    # Step 5: Branch path to match channels
    branch_path_output = SeparableConv2D(filters=96, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Step 6: Fuse both paths
    fused_output = Add()([main_path_output, branch_path_output])

    # Step 7: Flatten the output
    flatten_layer = Flatten()(fused_output)

    # Step 8: Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Step 9: Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Step 10: Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model