import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Add

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def main_path(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        max_pooling = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)
        return max_pooling

    def branch_path(input_tensor):
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=128, activation='relu')(avg_pool)
        dense2 = Dense(units=128, activation='relu')(dense1)
        channel_weights = Dense(units=128, activation='linear')(dense2)
        channel_weights = Reshape(target_shape=(128, 1))(channel_weights)
        weighted_input = Multiply()([input_tensor, channel_weights])
        return weighted_input

    main_output = main_path(input_tensor=input_layer)
    branch_output = branch_path(input_tensor=input_layer)
    adding_layer = Add()([main_output, branch_output])
    flatten_layer = Flatten()(adding_layer)
    output_layer1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer2 = Dense(units=10, activation='softmax')(output_layer1)

    model = keras.Model(inputs=input_layer, outputs=output_layer2)

    return model