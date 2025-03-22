import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense

def dl_model():
    # Define input layer
    input_layer = Input(shape=(28, 28, 1))

    # Define main path
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',activation='relu')(input_layer)
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',activation='relu')(main_path)
    main_path = Add()([main_path, input_layer])

    # Define branch path
    branch_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',activation='relu')(input_layer)

    # Combine main and branch paths
    output_layer = Add()([main_path, branch_path])

    # Flatten and classify output
    flatten_layer = Flatten()(output_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create and return model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model