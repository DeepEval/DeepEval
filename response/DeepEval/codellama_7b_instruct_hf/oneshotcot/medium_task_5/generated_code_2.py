import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Main path
    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(max_pooling1)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)
    main_path = Concatenate()([max_pooling1, max_pooling2])

    # Branch path
    input_layer_branch = Input(shape=(32, 32, 3))
    conv_branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer_branch)
    max_pooling_branch = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_branch)
    branch_path = max_pooling_branch

    # Combine main and branch paths
    combined_path = Concatenate()([main_path, branch_path])

    # Flatten and project onto probability distribution
    flatten = Flatten()(combined_path)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # Define and return the model
    model = keras.Model(inputs=[input_layer, input_layer_branch], outputs=dense2)
    return model