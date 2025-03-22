import keras
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Reshape
from keras.models import Model


def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    block_1 = Sequential()
    block_1.add(GlobalAveragePooling2D())
    block_1.add(Dense(64, activation='relu'))
    block_1.add(Flatten())

    block_2 = Sequential()
    block_2.add(Conv2D(64, (3, 3), activation='relu'))
    block_2.add(MaxPooling2D((2, 2)))
    block_2.add(Conv2D(64, (3, 3), activation='relu'))
    block_2.add(MaxPooling2D((2, 2)))

    main_path = block_1(input_layer)
    branch_path = block_2(main_path)
    fused_output = Add()([main_path, branch_path])
    flattened_output = Flatten()(fused_output)
    output_layer = Dense(10, activation='softmax')(flattened_output)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model