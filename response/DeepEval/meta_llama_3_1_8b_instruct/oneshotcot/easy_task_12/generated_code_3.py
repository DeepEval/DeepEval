import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, SeparableConv2D, Add

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    block_output1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    separable_conv1 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block_output1)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(separable_conv1)

    block_output2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling1)
    separable_conv2 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block_output2)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(separable_conv2)

    # Define the branch path
    branch_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(max_pooling2)

    # Sum the outputs of the main and branch paths
    summed_output = Add()([max_pooling2, branch_output])

    bath_norm = BatchNormalization()(summed_output)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model