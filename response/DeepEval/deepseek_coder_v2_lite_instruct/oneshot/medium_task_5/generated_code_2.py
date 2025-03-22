import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_conv1)
    main_pool = MaxPooling2D(pool_size=(2, 2), strides=2)(main_conv2)

    # Branch path
    branch_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_pool = MaxPooling2D(pool_size=(2, 2), strides=2)(branch_conv)

    # Addition of main path and branch path outputs
    combined = Add()([main_pool, branch_pool])

    # Flatten the combined output
    flatten_layer = Flatten()(combined)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model