import keras
from keras.layers import Input, Conv2D, ReLU, DepthwiseConv2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def feature_block(input_tensor):
        x = ReLU()(input_tensor)
        x = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
        return Concatenate(axis=3)([input_tensor, x])

    main_path = input_layer
    for _ in range(3):
        main_path = feature_block(main_path)

    branch_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    merged_path = keras.layers.add([main_path, branch_path])

    flatten_layer = Flatten()(merged_path)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model