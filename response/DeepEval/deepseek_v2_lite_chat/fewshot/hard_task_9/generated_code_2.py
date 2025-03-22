import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Concatenate, AveragePooling2D, Lambda, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def branch_1(input_tensor):
        conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv

    def branch_2(input_tensor):
        conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
        return Concatenate()([conv, conv2])

    def branch_3(input_tensor):
        conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        return Concatenate()([conv, conv2, conv3])

    branch1_output = branch_1(input_tensor=input_layer)
    branch2_output = branch_2(input_tensor=input_layer)
    branch3_output = branch_3(input_tensor=input_layer)

    fused_output = Add()([branch1_output, branch2_output, branch3_output])
    fused_output = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(fused_output)

    output_layer = Flatten()(fused_output)
    output_layer = Dense(units=1024, activation='relu')(output_layer)
    output_layer = Dense(units=512, activation='relu')(output_layer)
    output_layer = Dense(units=10, activation='softmax')(output_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model