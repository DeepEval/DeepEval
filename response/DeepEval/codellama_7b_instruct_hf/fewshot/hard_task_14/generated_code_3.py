import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    main_path = GlobalAveragePooling2D()(input_layer)
    branch_path = Conv2D(64, (3, 3), padding='same')(input_layer)
    added_path = Add()([main_path, branch_path])
    fc1 = Dense(64, activation='relu')(added_path)
    fc2 = Dense(32, activation='relu')(fc1)
    output_layer = Dense(10, activation='softmax')(fc2)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model