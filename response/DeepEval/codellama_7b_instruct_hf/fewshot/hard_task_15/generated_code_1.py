import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Add
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = GlobalAveragePooling2D()(input_layer)
    main_path = Dense(units=128, activation='relu')(main_path)
    main_path = Reshape(target_shape=(32, 32, 128))(main_path)
    main_path = Dense(units=32, activation='relu')(main_path)

    # Branch path
    branch_path = input_layer

    # Addition of main and branch paths
    output_layer = Add()([main_path, branch_path])

    # Final output layer
    output_layer = Dense(units=10, activation='softmax')(output_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model