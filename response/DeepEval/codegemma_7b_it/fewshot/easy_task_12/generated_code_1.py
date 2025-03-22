import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, SeparableConv2D, ReLU

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # Main Path
    def block(input_tensor):
        conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_tensor)
        separable_conv = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
        max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(separable_conv)
        return max_pooling

    main_path_output = block(input_tensor)

    # Branch Path
    branch_path_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)

    # Merging Paths
    merged_output = Add()([main_path_output, branch_path_output])

    # Flatten and Fully Connected Layers
    flatten_layer = Flatten()(merged_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model