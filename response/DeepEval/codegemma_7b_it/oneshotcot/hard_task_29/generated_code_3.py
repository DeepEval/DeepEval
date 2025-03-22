import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():

    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # First block
    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu')(conv1)

    # Branch path
    conv_branch = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    # Combine outputs of main and branch path
    main_path_out = Add()([conv1, conv2])
    branch_path_out = Add()([conv_branch])

    # Second block
    # Max pooling layers
    max_pool1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(main_path_out)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(main_path_out)
    max_pool3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(main_path_out)

    # Concatenate outputs of max pooling layers
    concat_out = keras.layers.concatenate([max_pool1, max_pool2, max_pool3])

    # Flatten and fully connected layers
    flatten = Flatten()(concat_out)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model