import keras
from keras.layers import Input, Conv2D, Concatenate, AveragePooling2D, Dense, Dropout

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def branch_1(input_tensor):
        x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        x = Dropout(0.2)(x)  # apply dropout to mitigate overfitting
        return x

    def branch_2(input_tensor):
        x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = Dropout(0.2)(x)  # apply dropout to mitigate overfitting
        return x

    def branch_3(input_tensor):
        x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = Dropout(0.2)(x)  # apply dropout to mitigate overfitting
        return x

    def branch_4(input_tensor):
        x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
        x = Dropout(0.2)(x)  # apply dropout to mitigate overfitting
        return x

    branch1_output = branch_1(input_tensor=input_layer)
    branch2_output = branch_2(input_tensor=input_layer)
    branch3_output = branch_3(input_tensor=input_layer)
    branch4_output = branch_4(input_tensor=input_layer)

    concatenated_output = Concatenate()([branch1_output, branch2_output, branch3_output, branch4_output])

    x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated_output)
    x = Flatten()(x)
    x = Dense(units=64, activation='relu')(x)
    x = Dense(units=32, activation='relu')(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model