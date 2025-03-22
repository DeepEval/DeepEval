import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    # Main path
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(main_path)
    main_path = Flatten()(main_path)
    main_path = Dense(units=10, activation='softmax')(main_path)
    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_path = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(branch_path)
    branch_path = Flatten()(branch_path)
    branch_path = Dense(units=10, activation='softmax')(branch_path)
    # Merge main and branch paths
    output = Add()([main_path, branch_path])
    # Output
    output = Flatten()(output)
    output = Dense(units=10, activation='softmax')(output)
    model = keras.Model(inputs=input_layer, outputs=output)
    return model