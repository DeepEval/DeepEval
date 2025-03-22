import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Flatten, Dense, Reshape, Add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    main_path = GlobalAveragePooling2D()(input_layer)
    main_path = Dense(units=32, activation='relu')(main_path)
    main_path = Dense(units=3, activation='relu')(main_path)
    main_path = Reshape((32, 32, 3))(main_path)

    # Branch Path
    branch_path = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Add Outputs
    output = Add()([main_path, branch_path])

    # Classification Layers
    output = Flatten()(output)
    output = Dense(units=128, activation='relu')(output)
    output = Dense(units=64, activation='relu')(output)
    output_layer = Dense(units=10, activation='softmax')(output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model