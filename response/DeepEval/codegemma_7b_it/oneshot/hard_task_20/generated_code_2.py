import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, add

def dl_model():

    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    def feature_extraction(input_tensor, kernel_size):
        conv = Conv2D(filters=64, kernel_size=kernel_size, strides=(1, 1), padding='same', activation='relu')(input_tensor)
        max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)
        return max_pooling

    # Split the input into three groups
    group1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Process each group with different kernel sizes in the main path
    output1 = feature_extraction(group1[0], (1, 1))
    output2 = feature_extraction(group1[1], (3, 3))
    output3 = feature_extraction(group1[2], (5, 5))

    # Concatenate the outputs from the main path
    concat_main_path = Concatenate()([output1, output2, output3])

    # Branch path
    branch_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Align the number of output channels
    concat_branch_path = Concatenate()([branch_path, tf.zeros_like(branch_path)])

    # Combine the outputs of the main and branch paths
    fused_features = add([concat_main_path, concat_branch_path])

    # Batch normalization and flattening
    bath_norm = BatchNormalization()(fused_features)
    flatten_layer = Flatten()(bath_norm)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model