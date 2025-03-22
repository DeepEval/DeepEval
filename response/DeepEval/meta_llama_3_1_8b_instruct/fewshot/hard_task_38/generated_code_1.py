import keras
from keras.layers import Input, Conv2D, BatchNormalization, Add, Concatenate, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        batch_norm = BatchNormalization()(input_tensor)
        conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(batch_norm)
        output_tensor = Concatenate()([input_tensor, conv])
        return output_tensor

    main_path = input_layer
    for _ in range(3):
        main_path = block(main_path)

    branch_path = input_layer
    for _ in range(3):
        branch_path = block(branch_path)

    merge_layer = Concatenate()([main_path, branch_path])

    flatten_layer = Flatten()(merge_layer)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model