import keras
from keras.layers import Input, Conv2D, Add, GlobalAveragePooling2D, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def branch_1(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        output_tensor = conv2
        return output_tensor

    def branch_2(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv1)
        output_tensor = conv2
        return output_tensor

    branch1_output = branch_1(input_layer)
    branch2_output = branch_2(input_layer)

    adding_layer = Add()([branch1_output, branch2_output])

    global_avg_pool = GlobalAveragePooling2D()(adding_layer)

    dense1 = Dense(units=128, activation='relu')(global_avg_pool)
    attention_weights = Dense(units=10, activation='softmax')(dense1)

    weighted_output = Dense(units=10, activation='softmax')(global_avg_pool)

    final_output = Multiply()([weighted_output, attention_weights])
    final_output = Add()([final_output, weighted_output])

    output_layer = Dense(units=10, activation='softmax')(final_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model