import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, GlobalAveragePooling2D, Flatten, Dense, Dropout

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    def main_path(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv)
        conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv)
        output_tensor = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(Add()([conv1, conv2]))
        output_tensor = Dropout(0.5)(output_tensor)
        return output_tensor

    branch_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(input_layer)

    main_path_output = main_path(input_tensor=input_layer)
    adding_layer = Add()([main_path_output, branch_path])

    global_avg_pool = GlobalAveragePooling2D()(adding_layer)
    flatten_layer = Flatten()(global_avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model