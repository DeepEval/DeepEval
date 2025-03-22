import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    conv_main_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_main_2 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_main_1)

    # Branch path
    branch_path = input_layer

    # Merging both paths
    merged = Add()([conv_main_2, branch_path])

    # Flatten and fully connected layers
    flatten_layer = Flatten()(merged)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Constructing the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model