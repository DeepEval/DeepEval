import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Path 1
    path1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)

    # Path 2
    path2 = AveragePooling2D(pool_size=(2, 2))(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(path2)

    # Path 3
    path3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    path3_1 = Conv2D(filters=32, kernel_size=(1, 3), activation='relu')(path3)
    path3_2 = Conv2D(filters=32, kernel_size=(3, 1), activation='relu')(path3)
    path3 = Concatenate()([path3_1, path3_2])

    # Path 4
    path4 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    path4_1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(path4)
    path4_2 = Conv2D(filters=32, kernel_size=(1, 3), activation='relu')(path4_1)
    path4_3 = Conv2D(filters=32, kernel_size=(3, 1), activation='relu')(path4_1)
    path4 = Concatenate()([path4_2, path4_3])

    # Concatenate all paths
    output = Concatenate()([path1, path2, path3, path4])

    # Flatten and classify
    output = Flatten()(output)
    output = Dense(units=10, activation='softmax')(output)

    model = keras.Model(inputs=input_layer, outputs=output)

    return model