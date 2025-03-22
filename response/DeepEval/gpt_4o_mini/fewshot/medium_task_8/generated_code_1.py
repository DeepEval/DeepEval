import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Concatenate, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    split_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)
    group1 = split_groups[0]  # First group remains unchanged
    group2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_groups[1])  # Second group
    group3 = split_groups[2]  # Third group

    # Combine group2 and group3
    combined = Add()([group2, group3])
    main_path_output = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(combined)

    # Concatenate all three groups
    main_path_output = Concatenate()([group1, main_path_output, group3])

    # Branch Path
    branch_path_output = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Fuse the main path and branch path outputs
    fused_output = Add()([main_path_output, branch_path_output])

    # Final classification layers
    flatten_layer = Flatten()(fused_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model