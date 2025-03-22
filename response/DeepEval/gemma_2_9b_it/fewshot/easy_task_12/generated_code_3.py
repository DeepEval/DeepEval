import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, DepthwiseConv2D, ReLU

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main Path
    x = ReLU()(input_layer)
    x = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x = ReLU()(x)
    x = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    # Branch Path
    branch_x = ReLU()(input_layer)
    branch_x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_x)
    branch_path = branch_x

    # Concatenate and Flatten
    merged_tensor = Add()([main_path, branch_path])
    flatten_layer = Flatten()(merged_tensor)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model