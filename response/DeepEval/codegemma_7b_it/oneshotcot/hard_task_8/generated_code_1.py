import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape, Permute

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    primary_path = Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    primary_path = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(primary_path)
    primary_path = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(primary_path)

    branch_path = Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    branch_path = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depthwise_initializer='he_normal')(branch_path)
    branch_path = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(branch_path)

    block1_output = Concatenate()([primary_path, branch_path])

    # Block 2
    block1_shape = keras.backend.int_shape(block1_output)

    reshaped_features = Reshape((-1, block1_shape[3], block1_shape[1] * block1_shape[2]))(block1_output)

    permuted_features = Permute((0, 2, 1))(reshaped_features)

    reshuffled_features = Reshape((block1_shape[1], block1_shape[2], block1_shape[3]))(permuted_features)

    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(reshuffled_features)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model