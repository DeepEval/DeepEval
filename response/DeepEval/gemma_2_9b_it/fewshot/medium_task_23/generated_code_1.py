import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Path 1
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)

    # Path 2
    conv2_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    conv2_2 = Conv2D(filters=32, kernel_size=(1, 7), activation='relu')(conv2_1)
    conv2_3 = Conv2D(filters=32, kernel_size=(7, 1), activation='relu')(conv2_1)

    # Path 3
    conv3_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    conv3_2_1 = Conv2D(filters=32, kernel_size=(1, 7), activation='relu')(conv3_1)
    conv3_2_2 = Conv2D(filters=32, kernel_size=(7, 1), activation='relu')(conv3_1)
    conv3_3_1 = Conv2D(filters=32, kernel_size=(7, 1), activation='relu')(conv3_2_1)
    conv3_3_2 = Conv2D(filters=32, kernel_size=(1, 7), activation='relu')(conv3_2_2)

    # Path 4
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2)(input_layer)
    conv4_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(avg_pool)

    # Concatenate outputs
    merged = Concatenate()([conv1_1, conv2_3, conv3_3_2, conv4_1])

    # Flatten and classify
    flatten = Flatten()(merged)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model