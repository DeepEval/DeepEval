import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add

def dl_model():

    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    initial_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(initial_conv)

    branch2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(initial_conv)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = UpSampling2D(size=(2, 2))(branch2)

    branch3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(initial_conv)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = UpSampling2D(size=(2, 2))(branch3)

    main_path_output = Concatenate()([branch1, branch2, branch3])
    main_path_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path_output)

    # Branch path
    branch_path_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_path_output = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_path_conv)

    # Combine outputs
    combined_output = Add()([main_path_output, branch_path_output])
    combined_output = BatchNormalization()(combined_output)

    # Fully connected layers
    flatten_layer = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model definition
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model