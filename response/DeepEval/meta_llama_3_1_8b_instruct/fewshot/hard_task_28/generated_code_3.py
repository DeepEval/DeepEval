import keras
from keras.layers import Input, DepthwiseConv2D, BatchNormalization, Conv2D, Add, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = BatchNormalization()(main_path)
    main_path = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = BatchNormalization()(main_path)
    main_path = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path)

    # Branch path
    branch_path = input_layer

    # Combine the outputs of both paths
    combined = Add()([main_path, branch_path])

    # Flatten and process through fully connected layers
    flatten_layer = Flatten()(combined)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model