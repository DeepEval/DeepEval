import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense
def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',activation='relu')(conv1)
    main_path = Add()([conv1, conv2])
    branch_path = Input(shape=(28, 28, 1))
    adding_layer = Add()([main_path, branch_path])
    flatten_layer = Flatten()(adding_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    model = keras.Model(inputs=[input_layer, branch_path], outputs=output_layer)
    return model