import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    main_path = GlobalAveragePooling2D()(input_layer)
    main_path = Dense(units=32, activation='relu')(main_path)
    main_path = Dense(units=3, activation='relu')(main_path)
    main_path = Reshape(target_shape=(32, 32, 3))(main_path)

    branch_path = input_layer

    adding_layer = Add()([main_path, branch_path])

    output_layer = Dense(units=10, activation='softmax')(adding_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model