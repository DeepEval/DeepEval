import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, BatchNormalization, Flatten, Dense

def dl_model():
    
    # Construct the model
    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)

    def block1(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        output_tensor = Add()([path1, path2])
        return output_tensor

    block1_output = block1(input_tensor=max_pooling1)
    add_layer = Add()([block1_output, input_layer])
    bath_norm = BatchNormalization()(add_layer)
    flatten_layer = Flatten()(bath_norm)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(flatten_layer)

    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling2)
    max_pooling3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)

    def block2(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        output_tensor = Add()([path1, path2])
        return output_tensor

    block2_output = block2(input_tensor=max_pooling3)
    bath_norm2 = BatchNormalization()(block2_output)
    flatten_layer2 = Flatten()(bath_norm2)

    dense1 = Dense(units=128, activation='relu')(flatten_layer2)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model