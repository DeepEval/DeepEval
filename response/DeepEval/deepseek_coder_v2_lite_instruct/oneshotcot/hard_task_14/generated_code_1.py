import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Add, Reshape
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    conv_main = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv_main = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv_main)
    global_avg_pool = GlobalAveragePooling2D()(conv_main)
    dense1 = Dense(units=64, activation='relu')(global_avg_pool)
    dense2 = Dense(units=32, activation='relu')(dense1)
    weights = Reshape((1, 1, 32))(dense2)
    main_path_output = Add()([conv_main * weights, conv_main])

    # Branch path
    conv_branch = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    branch_path_output = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='relu')(conv_branch)

    # Combine both paths
    combined_output = Add()([main_path_output, branch_path_output])

    # Flatten and fully connected layers
    flatten = keras.layers.Flatten()(combined_output)
    dense3 = Dense(units=128, activation='relu')(flatten)
    dense4 = Dense(units=64, activation='relu')(dense3)
    output_layer = Dense(units=10, activation='softmax')(dense4)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model