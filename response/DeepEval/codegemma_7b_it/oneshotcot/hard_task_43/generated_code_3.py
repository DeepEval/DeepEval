import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    path1_avg_pool_1x1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    path2_avg_pool_2x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    path3_avg_pool_4x4 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)

    # Flatten the outputs of the three parallel paths
    path1_avg_pool_1x1 = Flatten()(path1_avg_pool_1x1)
    path2_avg_pool_2x2 = Flatten()(path2_avg_pool_2x2)
    path3_avg_pool_4x4 = Flatten()(path3_avg_pool_4x4)

    # Concatenate the outputs of the three parallel paths
    block1_output = Concatenate()([path1_avg_pool_1x1, path2_avg_pool_2x2, path3_avg_pool_4x4])

    # Fully connected layer between Block 1 and Block 2
    block1_fc = Dense(units=128, activation='relu')(block1_output)

    # Reshape the output from Block 1 for compatibility with Block 2
    block1_output_reshaped = Reshape((1, 1, 128))(block1_fc)

    # Block 2
    path1_branch_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1_output_reshaped)
    path2_branch_3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1_output_reshaped)

    path3_branch_1x7_7x1 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(block1_output_reshaped)
    path3_branch_7x1_1x7 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3_branch_1x7_7x1)

    path4_branch_3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1_output_reshaped)

    # Average pooling for all branches
    path1_avg_pool_branch_1x1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(path1_branch_1x1)
    path2_avg_pool_branch_3x3 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(path2_branch_3x3)
    path3_avg_pool_branch_7x1_1x7 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(path3_branch_7x1_1x7)
    path4_avg_pool_branch_3x3 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(path4_branch_3x3)

    # Concatenate the outputs of all branches in Block 2
    block2_output = Concatenate()([path1_avg_pool_branch_1x1, path2_avg_pool_branch_3x3, path3_avg_pool_branch_7x1_1x7, path4_avg_pool_branch_3x3])

    # Flatten the output of Block 2
    block2_output = Flatten()(block2_output)

    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(block2_output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model