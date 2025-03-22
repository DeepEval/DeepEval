import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add

def dl_model():     
    input_layer = Input(shape=(28, 28, 1))

    # Branch 1
    branch_1_conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_1_conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_1_conv1)
    branch_1_conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_1_conv2)

    # Branch 2
    branch_2_conv1 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_2_conv2 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch_2_conv1)
    branch_2_conv3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch_2_conv2)

    # Branch 3
    branch_3_conv1 = Conv2D(filters=32, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_3_conv2 = Conv2D(filters=32, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(branch_3_conv1)
    branch_3_conv3 = Conv2D(filters=32, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(branch_3_conv2)

    # Branch 4
    branch_4_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine outputs from branches
    combined_outputs = Add()([branch_1_conv3, branch_2_conv3, branch_3_conv3, branch_4_conv])

    # Block 2
    block_2_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(combined_outputs)
    block_2_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block_2_conv1)
    block_2_conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(block_2_conv2)

    # Flatten and fully connected layers
    flatten_layer = Flatten()(block_2_conv3)
    dense = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense)

    # Model creation
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model